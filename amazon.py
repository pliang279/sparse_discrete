import json
import gzip
import numpy as np
import math
import os
import random
import pdb
import sys
import h5py
import pickle
from collections import Counter
from tqdm import tqdm, trange
from absl import app
from absl import flags
import pandas as pd

from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import *
from torch_yogi import Yogi
from torch_sgd import SGD

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', 'MF', 'model name')   # MF    mdMF    sparseMF    NCF     mdNCF   sparseNCF
flags.DEFINE_integer('num_epoch', 50, '')
flags.DEFINE_integer('batch_size', 10000, '')
flags.DEFINE_string('optimizer', 'yogi', '')
flags.DEFINE_float('lr', 0.1, '')
flags.DEFINE_integer('latent_dim', 16, '')
flags.DEFINE_bool('load', False, '')
flags.DEFINE_bool('test', False, '')
flags.DEFINE_string('device', 'cuda', '')

# scheduler
flags.DEFINE_integer('step_size', 50000, '')
flags.DEFINE_float('gamma', 0.5, '')

flags.DEFINE_string('dataset', 'amazon', '')

flags.DEFINE_integer('user_anchors', 20, '')
flags.DEFINE_integer('item_anchors', 20, '')
flags.DEFINE_float('lda1', 1e-2, '')
flags.DEFINE_float('lda2s', 0.0, '')
flags.DEFINE_integer('zerofor', 5, '')
flags.DEFINE_float('lda2e', 1e-5, '')

# for dynamic sparse embeddings
flags.DEFINE_bool('dynamic', False, '')
flags.DEFINE_integer('init_anchors', 10, '')
flags.DEFINE_integer('delta', 2, '')

# for md embeddings
flags.DEFINE_integer('base_dim', 32, '')
flags.DEFINE_float('temperature', 0.6, '')
flags.DEFINE_integer('k', 8, '')
flags.DEFINE_bool('round_dims', True, '')

# these below are set automatically by the code
flags.DEFINE_bool('sparse', False, '')
flags.DEFINE_bool('full_reg', False, '')
flags.DEFINE_bool('prune', False, '')
flags.DEFINE_float('prune_lda', 0.0, '')
flags.DEFINE_bool('md', False, '')
flags.DEFINE_integer('n_users', 0, '')
flags.DEFINE_integer('m_items', 0, '')
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
	#   if other not in data_train:
	#       freq_counter[other] = 0
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
	if FLAGS.sparse:
		print("size of user T: {} x {} = {}, nnz in user T: {}".format(FLAGS.n_users, model.user_anchors, FLAGS.n_users*model.user_anchors, nnz_user_T))
		print("size of item T: {} x {} = {}, nnz in item T: {}".format(FLAGS.m_items, model.item_anchors, FLAGS.m_items*model.item_anchors, nnz_item_T))
		print("size of user A: {} x {} = {}".format(model.user_anchors, FLAGS.latent_dim, model.user_anchors*FLAGS.latent_dim))
		print("size of item A: {} x {} = {}".format(model.item_anchors, FLAGS.latent_dim, model.item_anchors*FLAGS.latent_dim))
		print ('total nnz params:', nnz_user_T+nnz_item_T+model.user_anchors*FLAGS.latent_dim+model.item_anchors*FLAGS.latent_dim)
	elif FLAGS.full_reg:
		print ('embedding_user size: {} x {} = {}'.format(FLAGS.n_users, FLAGS.latent_dim, FLAGS.n_users*FLAGS.latent_dim))
		print ('embedding_item size: {} x {} = {}'.format(FLAGS.m_items, FLAGS.latent_dim, FLAGS.m_items*FLAGS.latent_dim))
		print ('total nnz params:', nnz_user_T+nnz_item_T)
	return

def compute_equation(loss, model, curr_lda2):
	nnz_user_T, nnz_item_T = model.get_nnz()
	nnz_total = curr_lda2*(nnz_user_T+nnz_item_T)
	anchors_total = (FLAGS.lda1-curr_lda2)*(model.user_anchors+model.item_anchors)
	print("Average loss {}".format(loss))
	print("Nnz users: {}, items: {}, weighted: {}".format(nnz_user_T, nnz_item_T, nnz_total))
	print("Num anchors users: {}, items: {}, weighted: {}".format(model.user_anchors, model.item_anchors, anchors_total))
	print("Overall loss: {}".format(loss+nnz_total+anchors_total))
	total_loss = loss + nnz_total + anchors_total
	return total_loss

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
				  {'params': sparse_params, 'regularization': (FLAGS.lda2s, 0.0)}]

	# optimizer = Adam(model.parameters(), lr=FLAGS.lr, amsgrad=True)
	optimizer = Yogi(params_opt, lr=FLAGS.lr)
	# optimizer = SGD(params_opt, lr=FLAGS.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=FLAGS.step_size, gamma=FLAGS.gamma)    # 100000, 0.5. s2: 200000, 0.5. s3: 500000, 0.5.
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=FLAGS.num_epoch)
	if FLAGS.sparse:
		initial_lda2 = FLAGS.lda2s
		final_lda2 = FLAGS.lda2e
		step = (final_lda2-initial_lda2) / float(FLAGS.num_epoch)
		lda2_schedule = [initial_lda2] * FLAGS.zerofor + [initial_lda2+i*step for i in range(FLAGS.num_epoch)]
	else:
		lda2_schedule = [0.0]*FLAGS.num_epoch

	model.zero_grad()

	total_losses = []
	best_val_loss = math.inf
	for epoch in range(FLAGS.num_epoch):
		model.train()
		total_train_loss = 0
		curr_lda2 = lda2_schedule[epoch]
		print ('curr epoch %d, l1 penalty: %s' % (epoch, str(curr_lda2)))
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
				print_nnz(model)
				print (torch.min(model_outputs), torch.min(ratings))
				print (torch.max(model_outputs), torch.max(ratings))
				print (torch.mean(model_outputs), torch.mean(ratings))

			# if iteration == 2000: break

		curr_val_loss = calculate_val_loss(model, val_dataloader)
		curr_train_loss = total_train_loss/iteration
		print("Training loss {}".format(curr_train_loss))
		print("Val loss {}".format(curr_val_loss))
		print_nnz(model)

		total_loss = compute_equation(curr_val_loss, model, curr_lda2)
		total_losses.append(total_loss)

		update_anchors(model, total_losses)

		if total_loss < best_val_loss:  # either take min total_losses or curr_val_loss
			best_val_loss = total_loss

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'num_anchors': (model.user_anchors, model.item_anchors),
				'curr_lda2': curr_lda2,
			}
			torch.save(checkpoint, FLAGS.model_path + 'checkpoint' + str(epoch) + '.pt')

		sys.stdout.flush()

def train(train_dataset, val_dataset, model):
	train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=FLAGS.batch_size)
	val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=8, batch_size=FLAGS.batch_size)
	'''
	params = list(model.parameters())
	if FLAGS.sparse:
		params.remove(model.user_T)
		params.remove(model.item_T)
		params_opt = [{'params': params},
					  {'params': [model.user_T, model.item_T], 'regularization': (FLAGS.lda2s, 0.0)}]
	elif FLAGS.full_reg:
		params.remove(model.embedding_user)
		params.remove(model.embedding_item)
		params_opt = [{'params': params},
					  {'params': [model.embedding_user, model.embedding_item], 'regularization': (FLAGS.lda2s, 0.0)}]
	else:
		params_opt = params
	'''

	if FLAGS.sparse:
		sparse_params = []
		nonsparse_params = []
		for name, param in model.named_parameters():
			if 'user_T' in name or 'item_T' in name:
				sparse_params.append(param)
			else:
				nonsparse_params.append(param)
		
		params_opt = [{'params': nonsparse_params},
					  {'params': sparse_params, 'regularization': (FLAGS.lda2s, 0.0)}]
	if FLAGS.sparse:
		sparse_params = []
		nonsparse_params = []
		for name, param in model.named_parameters():
			if 'embedding_user' in name or 'embedding_item' in name:
				sparse_params.append(param)
			else:
				nonsparse_params.append(param)
		
		params_opt = [{'params': nonsparse_params},
					  {'params': sparse_params, 'regularization': (FLAGS.lda2s, 0.0)}]
	else:
		params_opt = list(model.parameters())

	# optimizer = Adam(model.parameters(), lr=FLAGS.lr, amsgrad=True)
	optimizer = Yogi(params_opt, lr=FLAGS.lr)
	# optimizer = SGD(params_opt, lr=FLAGS.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=FLAGS.step_size, gamma=FLAGS.gamma)    # 100000, 0.5. s2: 200000, 0.5. s3: 500000, 0.5.
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=FLAGS.num_epoch)
	if FLAGS.sparse or FLAGS.full_reg:
		initial_lda2 = FLAGS.lda2s
		final_lda2 = FLAGS.lda2e
		step = (final_lda2-initial_lda2) / float(FLAGS.num_epoch)
		lda2_schedule = [initial_lda2] * FLAGS.zerofor + [initial_lda2+i*step for i in range(FLAGS.num_epoch)]
	else:
		lda2_schedule = [0.0]*FLAGS.num_epoch

	model.zero_grad()

	best_val_loss = math.inf
	for epoch in range(FLAGS.num_epoch):
		model.train()
		total_train_loss = 0
		curr_lda2 = lda2_schedule[epoch]
		print ('curr epoch %d, l1 penalty: %s' % (epoch, str(curr_lda2)))
		for iteration, batch in enumerate(tqdm(train_dataloader)):
			users, movies, ratings = batch
			users = users.to(FLAGS.device)
			movies = movies.to(FLAGS.device)
			ratings = ratings.to(FLAGS.device)
			
			model_outputs = model.forward(users, movies)
			train_loss = torch.mean((model_outputs-ratings)**2)

			# pdb.set_trace()

			train_loss.backward()
			optimizer.step()
			scheduler.step()
			model.zero_grad()
			curr_lr = scheduler.get_lr()[0]

			total_train_loss += train_loss.item()
			curr_train_loss = total_train_loss/(iteration+1)

			if iteration % 1 == 0:
				print('curr lr:', curr_lr)
				print("Training loss {}".format(curr_train_loss))
				if FLAGS.sparse or FLAGS.full_reg:
					print_nnz(model)
				if FLAGS.md:
					print ('embedding_user dim: {}'.format(FLAGS.total_user_dim))
					print ('embedding_item dim: {}'.format(FLAGS.total_item_dim))
					print ('total params:', FLAGS.total_user_dim + FLAGS.total_item_dim)

				print (torch.min(model_outputs), torch.min(ratings))
				print (torch.max(model_outputs), torch.max(ratings))
				print (torch.mean(model_outputs), torch.mean(ratings))

		curr_val_loss = calculate_val_loss(model, val_dataloader)
		curr_train_loss = total_train_loss/iteration
		print("Training loss {}".format(curr_train_loss))
		print("Val loss {}".format(curr_val_loss))
		if FLAGS.sparse or FLAGS.full_reg:
			print_nnz(model)
		if FLAGS.md:
			print ('embedding_user dim: {}'.format(FLAGS.total_user_dim))
			print ('embedding_item dim: {}'.format(FLAGS.total_item_dim))
			print ('total params:', FLAGS.total_user_dim + FLAGS.total_item_dim)

		if True: #curr_val_loss < best_val_loss:
			best_val_loss = curr_val_loss

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}
			torch.save(checkpoint, FLAGS.model_path + 'checkpoint' + str(epoch) + '.pt')

		sys.stdout.flush()

def clip(p, lda):
	return (p.data.abs() > lda).type(p.data.dtype) * p.data

def prune(model, curr_lda):
	if FLAGS.sparse:
		model.user_T.data = clip(model.user_T, curr_lda)
		model.item_T.data = clip(model.item_T, curr_lda)
	elif FLAGS.full_reg:
		model.embedding_user.data = clip(model.embedding_user, curr_lda)
		model.embedding_item.data = clip(model.embedding_item, curr_lda)
	return

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
		(user_anchors, item_anchors) = checkpoint['num_anchors']
		curr_lda2 = checkpoint['curr_lda2']
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
		compute_equation(total_loss, model, curr_lda2)

	return total_loss

def load_weights(model, movies_rev_map):
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

	MOVIES_CSV_FILE = 'ml-25m/movies.csv'
	movies = pd.read_csv(MOVIES_CSV_FILE,
						  sep=',', 
						  encoding='latin-1', 
						  usecols=['movieId', 'title', 'genres'])
	movie_list = dict()
	for (movie_id, movie_name, movie_genre) in zip(movies['movieId'].values, movies['title'].values, movies['genres'].values):
		movie_list[movie_id] = (movie_name, movie_genre)

	if FLAGS.dynamic:
		(user_anchors, item_anchors) = checkpoint['num_anchors']
		curr_lda2 = checkpoint['curr_lda2']
		model.user_anchors = user_anchors
		model.item_anchors = item_anchors
		item_T = model.all_item_T[:,:item_anchors]
		item_A = model.all_item_A[:item_anchors]

	for item_index in range(item_anchors):
		curr_frac = item_T[:,item_index] / torch.sum(item_T, axis=1)
		curr_frac[curr_frac != curr_frac] = 0
		sorted_fracs, indices = torch.sort(curr_frac, descending=True)
		indices = indices.cpu().numpy()
		nnz = (sorted_fracs!=0).sum()
		print (indices, nnz)
		top_indices = [movies_rev_map[idx] for idx in indices[:nnz]]
		top_movies = [movie_list[idx+1] for idx in top_indices] # code subtracted 1 from movie indices at the start! so plus 1 back
		# print ('top_indices:', top_indices)
		# print ('top_movies:', top_movies)

		genres = [genre for (_,genre) in top_movies]
		# pdb.set_trace()
		purity = compute_purity(genres)
		print ('purity:', purity)

	pdb.set_trace()

def compute_purity(genres):
	total_num = len(genres)
	all_genres = set()
	for genre in genres:
		for each in genre.split('|'):
			all_genres.add(each)
	nums = []
	for genre in all_genres:
		num = 0
		for movie in genres:
			if genre in movie:
				num += 1
		nums.append((float(num/total_num),genre))
	nums = sorted(nums, key=lambda x: x[0])
	return nums[-1]

def main(argv):
	cuda = torch.cuda.is_available()
	FLAGS.device = torch.device("cuda" if cuda else "cpu") # 'cpu'

	if FLAGS.dynamic and FLAGS.dataset == 'amazon':
		FLAGS.model_path = 'saved_models_dynamic_amazon/' + FLAGS.model_path
	elif FLAGS.dataset == 'amazon':
		FLAGS.model_path = 'saved_models_amazon/' + FLAGS.model_path
	else:
		print ('wrong flags')
		assert False

	FLAGS.sparse = 'sparse' in FLAGS.model_path
	FLAGS.full_reg = 'full_reg' in FLAGS.model_path
	FLAGS.md = 'md' in FLAGS.model_path

	if FLAGS.dynamic:
		FLAGS.model_path += '_lda1%s_lda2s%s_zerofor%s_lda2e%s_d%d_i%d_s%d_g%s_dynamic' %(str(FLAGS.lda1), str(FLAGS.lda2s), str(FLAGS.zerofor), str(FLAGS.lda2e), FLAGS.delta, FLAGS.init_anchors, FLAGS.step_size, str(FLAGS.gamma))
	elif FLAGS.sparse:
		FLAGS.model_path += '_ua%d_ia%d_lda2s%s_zerofor%s_lda2e%s_s%d_g%s' %(FLAGS.user_anchors, FLAGS.item_anchors, str(FLAGS.lda2s), str(FLAGS.zerofor), str(FLAGS.lda2e), FLAGS.step_size, str(FLAGS.gamma))
	elif FLAGS.md:
		FLAGS.model_path += '_base%d_temp%0.1f_k%d_s%d_g%s' %(FLAGS.base_dim, FLAGS.temperature, FLAGS.k, FLAGS.step_size, str(FLAGS.gamma))
	else:
		FLAGS.model_path += '_s%d_g%s' %(FLAGS.step_size, str(FLAGS.gamma))

	FLAGS.model_path += '_dim%d_split0.9' %(FLAGS.latent_dim)
	if FLAGS.prune:
		FLAGS.model_path += '_prune%s' %(str(FLAGS.prune_lda))
	FLAGS.model_path += '/'

	print (FLAGS.model_path)

	if not os.path.exists(FLAGS.model_path):
		if FLAGS.test:
			print ('FLAGS.model_path folder does not exist')
			assert False
		else:
			os.mkdir(FLAGS.model_path)

	if FLAGS.dataset == 'amazon':
		hf = h5py.File('amazon_data/saved_amazon_data_filtered5.h5', 'r') # _filtered5
		print ('opened data')
		num = 100000000000
		users = np.array(hf.get('users')) #[:num]
		movies = np.array(hf.get('items')) #[:num]
		ratings = np.array(hf.get('ratings')) #[:num]
		print ('loaded data')
		hf.close()

		RNG_SEED = 1446557
		np.random.seed(RNG_SEED)
		rng_state = np.random.get_state()
		np.random.shuffle(users)
		np.random.set_state(rng_state)
		np.random.shuffle(movies)
		np.random.set_state(rng_state)
		np.random.shuffle(ratings)

		FLAGS.n_users = np.max(users)+1 # 15167257 # len(user_forward_map)
		FLAGS.m_items = np.max(movies)+1 # 43531850 # len(item_forward_map)

	print ('Users:', users, ', shape =', users.shape, 'num =', FLAGS.n_users)
	print ('Items:', movies, ', shape =', movies.shape, 'num =', FLAGS.m_items)
	print ('Ratings:', ratings, ', shape =', ratings.shape)

	# pdb.set_trace()
	train_prop = 0.9
	val_prop = 0.1
	test_prop = 0.0
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
	# test_dataset = MovieDataset(users_test, movies_test, ratings_test)

	if FLAGS.dynamic:
		model = DynamicMFModel(FLAGS)
	elif 'NCFModel' in FLAGS.model_path:
		model = NCFModel(FLAGS)
	elif 'MF' in FLAGS.model_path:
		model = MFModel(FLAGS)

	model.to(FLAGS.device)

	if FLAGS.load:
		load_weights(model, movies_rev_map)
	elif FLAGS.test:
		test(val_dataset, model)
	elif FLAGS.dynamic:
		dynamic_train(train_dataset, val_dataset, model)
	elif FLAGS.prune:
		train_prune(train_dataset, val_dataset, model)
	else:
		train(train_dataset, val_dataset, model)


if __name__ == "__main__":
	app.run(main)


