import math
import os
import pdb
import sys
from collections import Counter
from tqdm import tqdm, trange
from absl import app
from absl import flags
import pandas as pd
import matplotlib.pyplot as plt
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import *
from torch_yogi import Yogi

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', 'MF', 'model name')	# MF 	mdMF 	sparseMF 	NCF 	mdNCF 	sparseNCF
flags.DEFINE_integer('num_epoch', 50, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_string('optimizer', 'adam', '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_integer('latent_dim', 16, '')
flags.DEFINE_bool('test', False, '')
flags.DEFINE_string('device', 'cuda', '')

flags.DEFINE_string('dataset', '1m', '')

flags.DEFINE_bool('dynamic', False, '')
flags.DEFINE_bool('sparse', False, '')
flags.DEFINE_bool('md', False, '')

flags.DEFINE_integer('n_users', 0, '')
flags.DEFINE_integer('m_items', 0, '')

flags.DEFINE_integer('user_anchors', 20, '')
flags.DEFINE_integer('item_anchors', 20, '')
flags.DEFINE_float('lda1', 1e-2, '')
flags.DEFINE_float('lda2', 1e-4, '')

flags.DEFINE_integer('init_anchors', 10, '')
flags.DEFINE_integer('delta', 1, '')

# for md embeddings
flags.DEFINE_integer('base_dim', 16, '')
flags.DEFINE_float('temperature', 0.6, '')
flags.DEFINE_integer('k', 8, '')
flags.DEFINE_bool('round_dims', True, '')

flags.DEFINE_integer('md_nums_user', 0, '')
flags.DEFINE_integer('md_dims_user', 0, '')
flags.DEFINE_integer('total_user_dim', 0, '')
flags.DEFINE_integer('md_nums_item', 0, '')
flags.DEFINE_integer('md_dims_item', 0, '')
flags.DEFINE_integer('total_item_dim', 0, '')
flags.DEFINE_integer('num_buckets_user', 0, '')
flags.DEFINE_integer('num_buckets_item', 0, '')


def init_md(data_train, data_val, data_test, total_indices):
	# add all indices (including not in train data), with freq = 1, indices start from 0
	data_train_padded = np.concatenate([np.array(data_train), np.array(range(total_indices))], axis=0)
	freq_counter = Counter(data_train_padded)
	# for other in tqdm():
	# 	if other not in data_train:
	# 		freq_counter[other] = 0
	freq_counter = freq_counter.most_common()
	freqs = torch.tensor(np.array([freq for (key, freq) in freq_counter]))
	freq_index = {}
	for index, (key, freq) in enumerate(freq_counter):
		freq_index[key] = index
	indices_train = torch.tensor(np.array([freq_index[key] for key in data_train]))
	indices_val = torch.tensor(np.array([freq_index[key] for key in data_val]))
	indices_test = torch.tensor(np.array([freq_index[key] for key in data_test]))

	tau = total_indices
	each_block = sum(freqs) / FLAGS.k
	nums, totals = [], []
	num, total = 0, 0
	index_to_bucket = {}
	bucket, bucket_id = 0, 0
	for index, freq in tqdm(enumerate(freqs)):
		total += freq
		num += 1
		index_to_bucket[index] = bucket, bucket_id
		if total >= each_block:
			nums.append(num)
			totals.append(total)
			total, num = 0, 0
			if bucket < FLAGS.k-1:
				bucket += 1
				bucket_id = 0
			else:
				bucket_id += 1
		else:
			bucket_id += 1

	remainder_nums = len(freqs) - sum(nums)
	remainder_totals = sum(freqs) - sum(totals)
	nums.append(remainder_nums)
	totals.append(remainder_totals)
	ps = torch.tensor([float(num)/float(tau) for num in nums])
	lda = FLAGS.base_dim*ps[0]**(FLAGS.temperature)
	dims = lda*ps**(-FLAGS.temperature)

	for i in tqdm(range(len(dims))):
		if i == 0:
			dims[i] = FLAGS.base_dim
		if dims[i] < 1 or torch.isnan(dims[i]):
			dims[i] = 1
	if FLAGS.round_dims:
		dims = 2 ** torch.round(torch.log2(dims.type(torch.float)))
	
	total_emb_dims = sum([a*d for (a,d) in zip(nums, dims)]).item()
	total_proj_dims = 0
	for dim in dims:
		if dim != FLAGS.latent_dim:
			total_proj_dims += dim.item()*FLAGS.latent_dim
	total_dim = total_emb_dims + total_proj_dims

	dims = [int(d) for d in dims.tolist()]
	indices_train = [index_to_bucket[key] for key in indices_train.data.numpy()]
	indices_val = [index_to_bucket[key] for key in indices_val.data.numpy()]
	indices_test = [index_to_bucket[key] for key in indices_test.data.numpy()]
	return nums, dims, total_dim, indices_train, indices_val, indices_test


class MovieDataset(Dataset):
	def __init__(self, users, movies, ratings):
		self.users = users
		self.movies = movies
		self.ratings = ratings

	def __len__(self):
		return len(self.users)

	def __getitem__(self, data_index):
		if FLAGS.md:
			bucket_user, bucket_id_user = self.users[data_index]
			bucket_item, bucket_id_item = self.movies[data_index]
			users = torch.tensor(bucket_user*FLAGS.num_buckets_user+bucket_id_user, dtype=torch.long)
			movies = torch.tensor(bucket_item*FLAGS.num_buckets_item+bucket_id_item, dtype=torch.long)
		else:
			users = torch.tensor(self.users[data_index], dtype=torch.long)
			movies = torch.tensor(self.movies[data_index], dtype=torch.long)
		ratings = torch.tensor(self.ratings[data_index], dtype=torch.float32)
		return users, movies, ratings

def calculate_val_loss(model, val_dataloader):
	model.eval()
	total_val_loss = 0
	with torch.no_grad():
		for iteration, batch in enumerate(tqdm(val_dataloader)):
			users, movies, ratings = batch
			users = users.to(FLAGS.device)
			movies = movies.to(FLAGS.device)
			ratings = ratings.to(FLAGS.device)
			model_outputs = model.forward(users, movies)
			val_loss = torch.mean((model_outputs-ratings)**2)
			total_val_loss += val_loss.item()

	total_val_loss = total_val_loss / len(val_dataloader)
	return total_val_loss

def print_nnz(model):
	nnz_user_T, nnz_item_T = model.get_nnz()
	print("size of user T: {} x {} = {}, nnz in user T: {}".format(FLAGS.n_users, model.user_anchors, FLAGS.n_users*model.user_anchors, nnz_user_T))
	print("size of item T: {} x {} = {}, nnz in item T: {}".format(FLAGS.m_items, model.item_anchors, FLAGS.m_items*model.item_anchors, nnz_item_T))
	print("size of user A: {} x {} = {}".format(model.user_anchors, FLAGS.latent_dim, model.user_anchors*FLAGS.latent_dim))
	print("size of item A: {} x {} = {}".format(model.item_anchors, FLAGS.latent_dim, model.item_anchors*FLAGS.latent_dim))
	print ('total nnz params:', nnz_user_T+nnz_item_T+model.user_anchors*FLAGS.latent_dim+model.item_anchors*FLAGS.latent_dim)
	return

def compute_equation(train_loss, model):
	nnz_user_T, nnz_item_T = model.get_nnz()
	nnz_total = FLAGS.lda2*(nnz_user_T+nnz_item_T)
	anchors_total = (FLAGS.lda1-FLAGS.lda2)*(model.user_anchors+model.item_anchors)
	print("Average loss {}".format(train_loss))
	print("Nnz users: {}, items: {}, weighted: {}".format(nnz_user_T, nnz_item_T, nnz_total))
	print("Num anchors users: {}, items: {}, weighted: {}".format(model.user_anchors, model.item_anchors, anchors_total))
	print("Overall loss: {}".format(train_loss+nnz_total+anchors_total))
	return train_loss, nnz_total, anchors_total

def improving(total_losses):
	if len(total_losses) < 3:
		return True
	else:
		return total_losses[-3] >= total_losses [-2] >= total_losses[-1]

def worsening(total_losses):
	if len(total_losses) < 3:
		return False
	else:
		return total_losses[-3] <= total_losses [-2] <= total_losses[-1]

def update_anchors(model, total_losses):
	print ('updating number of anchors based on loss counter:', total_losses)
	if improving(total_losses):
		print ('improving so adding anchors')
		model.expand_user()
		model.expand_item()
	elif worsening(total_losses):
		print ('worsening so reducing anchors')
		model.reduce_user()
		model.reduce_item()
	else:
		print ('neither so keeping same anchors')

def dynamic_train(train_dataset, val_dataset, model):
	train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=FLAGS.batch_size)
	val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=8, batch_size=FLAGS.batch_size)

	sparse_params = []
	nonsparse_params = []
	for name, param in model.named_parameters():
		if 'all_user_T' in name or 'all_item_T' in name:
			sparse_params.append(param)
		else:
			nonsparse_params.append(param)
	
	params_opt = [{'params': nonsparse_params},
			 	  {'params': sparse_params, 'regularization': (FLAGS.lda2, 0.0)}]

	optimizer = Yogi(params_opt, lr=FLAGS.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
	model.zero_grad()

	total_losses = []
	best_val_loss = math.inf
	for epoch in range(FLAGS.num_epoch):
		model.train()
		total_train_loss = 0
		for iteration, batch in enumerate(tqdm(train_dataloader)):
			users, movies, ratings = batch
			users = users.to(FLAGS.device)
			movies = movies.to(FLAGS.device)
			ratings = ratings.to(FLAGS.device)

			model_outputs = model.forward(users, movies)
			# pdb.set_trace()
			train_loss = torch.mean((model_outputs-ratings)**2)
			train_loss.backward()
			optimizer.step()
			scheduler.step()
			model.zero_grad()
			curr_lr = scheduler.get_lr()[0]

			total_train_loss += train_loss.item()
			curr_train_loss = total_train_loss/(iteration+1)

			if iteration % 1000 == 0:
				print('curr lr:', curr_lr)
				print("Training loss {}".format(curr_train_loss))
				print_nnz(model)
				print (torch.min(model_outputs), torch.min(ratings))
				print (torch.max(model_outputs), torch.max(ratings))
				print (torch.mean(model_outputs), torch.mean(ratings))

			# if iteration == 2000:	break

		curr_val_loss = calculate_val_loss(model, val_dataloader)
		curr_train_loss = total_train_loss/iteration
		print("Training loss {}".format(curr_train_loss))
		print("Val loss {}".format(curr_val_loss))
		print_nnz(model)

		loss, nnz_total, anchors_total = compute_equation(curr_val_loss, model)
		total_loss = loss + nnz_total + anchors_total
		total_losses.append(total_loss)

		update_anchors(model, total_losses)

		if total_loss < best_val_loss:	# either take min total_losses or curr_val_loss
			best_val_loss = total_loss

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'num_anchors': (model.user_anchors, model.item_anchors)
			}
			torch.save(checkpoint, FLAGS.model_path + 'checkpoint' + str(epoch) + '.pt')

		sys.stdout.flush()

def train(train_dataset, val_dataset, model):
	train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=FLAGS.batch_size)
	val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=8, batch_size=FLAGS.batch_size)

	params = list(model.parameters())
	if FLAGS.sparse:
		params.remove(model.user_T)
		params.remove(model.item_T)
		params_opt = [{'params': params},
					  {'params': [model.user_T, model.item_T], 'regularization': (FLAGS.lda2, 0.0)}]
	else:
		params_opt = params

	# optimizer = Adam(model.parameters(), lr=FLAGS.lr, amsgrad=True)
	optimizer = Yogi(params_opt, lr=FLAGS.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=FLAGS.num_epoch)

	model.zero_grad()

	best_val_loss = math.inf
	for epoch in range(FLAGS.num_epoch):
		model.train()
		total_train_loss = 0
		for iteration, batch in enumerate(tqdm(train_dataloader)):
			users, movies, ratings = batch
			users = users.to(FLAGS.device)
			movies = movies.to(FLAGS.device)
			ratings = ratings.to(FLAGS.device)
			
			model_outputs = model.forward(users, movies)
			train_loss = torch.mean((model_outputs-ratings)**2)

			train_loss.backward()
			optimizer.step()
			scheduler.step()
			model.zero_grad()
			curr_lr = scheduler.get_lr()[0]

			total_train_loss += train_loss.item()
			curr_train_loss = total_train_loss/(iteration+1)

			if iteration % 1000 == 0:
				print('curr lr:', curr_lr)
				print("Training loss {}".format(curr_train_loss))
				if FLAGS.sparse:
					print_nnz(model)
				if FLAGS.md:
					print ('embedding_user dim: {}'.format(FLAGS.total_user_dim))
					print ('embedding_item dim: {}'.format(FLAGS.total_item_dim))
					print ('total params:', FLAGS.total_user_dim + FLAGS.total_item_dim)

				# print (torch.min(model_outputs), torch.min(ratings))
				# print (torch.max(model_outputs), torch.max(ratings))
				# print (torch.mean(model_outputs), torch.mean(ratings))

		curr_val_loss = calculate_val_loss(model, val_dataloader)
		curr_train_loss = total_train_loss/iteration
		print("Training loss {}".format(curr_train_loss))
		print("Val loss {}".format(curr_val_loss))
		if FLAGS.sparse:
			print_nnz(model)

		if curr_val_loss < best_val_loss:
			best_val_loss = curr_val_loss

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}
			torch.save(checkpoint, FLAGS.model_path + 'checkpoint' + str(epoch) + '.pt')

		sys.stdout.flush()

def get_num_anchors(filename):
	anchors = []
	val_losses = []
	overall_losses = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			if 'Num anchors users' in line:
				anchors.append((int(line.split(',')[0].split(' ')[-1]), int(line.split(',')[1].split(' ')[-1])))
			elif 'Val loss' in line:
				val_losses.append(float(line.split(' ')[-1]))
			elif 'Overall loss' in line:
				overall_losses.append(float(line.split(' ')[-1]))
	# print (anchors, val_losses)
	# pdb.set_trace()
	anchor = anchors[val_losses.index(min(val_losses))]
	anchor = anchors[overall_losses.index(min(overall_losses))]
	print ('best val_loss:', val_losses.index(min(val_losses)), anchors[val_losses.index(min(val_losses))])
	print ('best overall_loss:', overall_losses.index(min(overall_losses)), anchors[overall_losses.index(min(overall_losses))])
	return anchor

def test(test_dataset, model):
	test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, batch_size=FLAGS.batch_size)

	all_checkpoints = []
	for file in os.listdir(FLAGS.model_path):
		if file.endswith(".pt"):
			all_checkpoints.append(os.path.join(FLAGS.model_path, file))

	all_checkpoints = sorted(all_checkpoints, key=lambda path: int(path[path.index('point')+5:-3]))
	checkpoint_path = all_checkpoints[-1]
	print ('loading from:', checkpoint_path)
	# pdb.set_trace()
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model'])
	model.to(FLAGS.device)
	model.eval()

	if FLAGS.dynamic:
		if 'num_anchors' in checkpoint:
			(user_anchors, item_anchors) = checkpoint['num_anchors']
		else:
			str_lda2 = '0.00008' if FLAGS.lda2 == 0.00008 else str(FLAGS.lda2)
			user_anchors, item_anchors = get_num_anchors('res_new1m/sparse_dynamic_%s_%s_d%d.txt'%(str(FLAGS.lda1), str_lda2, FLAGS.delta))
		model.user_anchors = user_anchors
		model.item_anchors = item_anchors

	if FLAGS.sparse:
		print_nnz(model)
	elif FLAGS.md:
		print ('embedding_user dim (plus proj): {}'.format(FLAGS.total_user_dim))
		print ('embedding_item dim (plus proj): {}'.format(FLAGS.total_item_dim))
		print ('total params:', FLAGS.total_user_dim + FLAGS.total_item_dim)
	else:
		print ('embedding_user shape: {} x {} = {}'.format(FLAGS.n_users, FLAGS.latent_dim, FLAGS.n_users*FLAGS.latent_dim))
		print ('embedding_item shape: {} x {} = {}'.format(FLAGS.m_items, FLAGS.latent_dim, FLAGS.m_items*FLAGS.latent_dim))
		print ('total params:', FLAGS.n_users*FLAGS.latent_dim+FLAGS.m_items*FLAGS.latent_dim)

	total_loss = 0
	with torch.no_grad():
		for iteration, batch in enumerate(tqdm(test_dataloader)):
			users, movies, ratings = batch
			users = users.to(FLAGS.device)
			movies = movies.to(FLAGS.device)
			ratings = ratings.to(FLAGS.device)
			model_outputs = model.forward(users, movies)
			loss = torch.mean((model_outputs-ratings)**2)
			total_loss += loss.item()

	total_loss = total_loss / len(test_dataloader)
	print("Test loss {}".format(total_loss))

	if FLAGS.dynamic:
		compute_equation(total_loss, model)

	return total_loss

def main(argv):
	cuda = torch.cuda.is_available()
	FLAGS.device = torch.device("cuda" if cuda else "cpu")
	
	if FLAGS.dynamic:
		FLAGS.model_path = 'saved_models_dynamic1m/' + FLAGS.model_path
	elif FLAGS.dataset == '1m':
		FLAGS.model_path = 'saved_models_full1m/' + FLAGS.model_path
	elif FLAGS.dataset == '25m':
		FLAGS.model_path = 'saved_models_25m/' + FLAGS.model_path

	FLAGS.sparse = 'sparse' in FLAGS.model_path
	FLAGS.md = 'md' in FLAGS.model_path

	if FLAGS.dynamic:
		FLAGS.model_path += '_lda1_%0.6f_lda2_%0.6f_d%d_i%d_dynamic' %(FLAGS.lda1, FLAGS.lda2, FLAGS.delta, FLAGS.init_anchors)
	elif FLAGS.sparse:
		FLAGS.model_path += '_ua%d_ia%d_lda2_%0.6f' %(FLAGS.user_anchors, FLAGS.item_anchors, FLAGS.lda2)
	elif FLAGS.md:
		FLAGS.model_path += '_base%d_temp%0.1f_k%d' %(FLAGS.base_dim, FLAGS.temperature, FLAGS.k)

	FLAGS.model_path += '_dim%d' %(FLAGS.latent_dim)
	FLAGS.model_path += '/'

	print (FLAGS.model_path)

	if not os.path.exists(FLAGS.model_path):
		if FLAGS.test:
			print ('FLAGS.model_path folder does not exist')
			assert False
		else:
			os.mkdir(FLAGS.model_path)

	if FLAGS.dataset == '1m':
		RATINGS_CSV_FILE = 'ml1m_ratings.csv'
		ratings = pd.read_csv(RATINGS_CSV_FILE,
							  sep='\t', 
							  encoding='latin-1', 
							  usecols=['userid', 'movieid', 'user_emb_id', 'movie_emb_id', 'rating'])
		user_id_header = 'userid'
		movie_id_header = 'movieid'
	elif FLAGS.dataset == '25m':
		RATINGS_CSV_FILE = 'ml-25m/ratings.csv'
		ratings = pd.read_csv(RATINGS_CSV_FILE,
							  sep=',', 
							  encoding='latin-1', 
							  usecols=['userId', 'movieId', 'rating', 'timestamp'])
		user_id_header = 'userId'
		movie_id_header = 'movieId'
	RNG_SEED = 1446557

	FLAGS.n_users = ratings[user_id_header].drop_duplicates().max()
	FLAGS.m_items = ratings[movie_id_header].drop_duplicates().max()
	print (len(ratings), 'ratings loaded.')

	shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
	users = shuffled_ratings[user_id_header].values - 1
	print ('Users:', users, ', shape =', users.shape)
	movies = shuffled_ratings[movie_id_header].values - 1
	print ('Movies:', movies, ', shape =', movies.shape)
	ratings = shuffled_ratings['rating'].values
	print ('Ratings:', ratings, ', shape =', ratings.shape)

	# pdb.set_trace()
	train_prop = 0.8
	val_prop = 0.1
	test_prop = 0.1
	users_train = users[:int(len(users)*train_prop)]
	movies_train = movies[:int(len(users)*train_prop)]
	ratings_train = ratings[:int(len(users)*train_prop)]

	users_val = users[int(len(users)*train_prop):int(len(users)*(train_prop+val_prop))]
	movies_val = movies[int(len(users)*train_prop):int(len(users)*(train_prop+val_prop))]
	ratings_val = ratings[int(len(users)*train_prop):int(len(users)*(train_prop+val_prop))]

	users_test = users[int(len(users)*(train_prop+val_prop)):]
	movies_test = movies[int(len(users)*(train_prop+val_prop)):]
	ratings_test = ratings[int(len(users)*(train_prop+val_prop)):]

	if FLAGS.md:
		# users_train = [1,5,4,4,3,3,6,3,4,4,1]
		# users_val = [1,4,2,3,5,6,0]
		# users_test = [5,5,4,6,7,2]
		# FLAGS.n_users = 8
		
		FLAGS.md_nums_user, FLAGS.md_dims_user, FLAGS.total_user_dim, \
		users_train, users_val, users_test \
		= init_md(users_train, users_val, users_test, FLAGS.n_users)
		FLAGS.md_nums_item, FLAGS.md_dims_item, FLAGS.total_item_dim, \
		movies_train, movies_val, movies_test \
		= init_md(movies_train, movies_val, movies_test, FLAGS.m_items)

		FLAGS.num_buckets_user = len(FLAGS.md_nums_user)
		FLAGS.num_buckets_item = len(FLAGS.md_nums_item)
		
	# pdb.set_trace()

	train_dataset = MovieDataset(users_train, movies_train, ratings_train)
	val_dataset = MovieDataset(users_val, movies_val, ratings_val)
	test_dataset = MovieDataset(users_test, movies_test, ratings_test)

	if FLAGS.dynamic:
		model = DynamicMFModel(FLAGS)
	elif 'NCFModel' in FLAGS.model_path:
		model = NCFModel(FLAGS)
	elif 'MF' in FLAGS.model_path:
		model = MFModel(FLAGS)

	model.to(FLAGS.device)

	if FLAGS.test:
		test(test_dataset, model)
	elif FLAGS.dynamic:
		dynamic_train(train_dataset, val_dataset, model)
	else:
		train(train_dataset, val_dataset, model)
	
def generate_grid():
	user_as = [1,2,3,5,8,10,15,20,30,50,80,100,120,150,200,300,500,800,1000,1500,2000,3000,4000,5000,6000]	# 6040
	item_as = [1,2,3,5,8,10,15,20,30,50,80,100,120,150,200,300,500,800,1000,1500,2000,3000,3500]			# 3952

	for user_a in user_as:
		for item_a in item_as:
			gpu = 0
			line = 'CUDA_VISIBLE_DEVICES=%d nohup python3 movielens.py --model_path sparseMF --latent_dim 16 --user_anchors %d --item_anchors %d --lda2 0.0001 --dataset 1m > res_full1m/sparse_ua%d_ia%d_0.0001.txt &' % (gpu,user_a,item_a,user_a,item_a)
			print (line)

if __name__ == "__main__":
	app.run(main)

