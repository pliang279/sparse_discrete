import math
import os
import pdb
import sys
from collections import Counter
from tqdm import tqdm, trange
import argparse
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

class MovieDataset(Dataset):
	def __init__(self, args, users, movies, ratings):
		self.args = args
		self.users = users
		self.movies = movies
		self.ratings = ratings

	def __len__(self):
		return len(self.users)

	def __getitem__(self, data_index):
		if self.args.md:
			bucket_user, bucket_id_user = self.users[data_index]
			bucket_item, bucket_id_item = self.movies[data_index]
			users = torch.tensor(bucket_user*self.args.num_buckets_user+bucket_id_user, dtype=torch.long)
			movies = torch.tensor(bucket_item*self.args.num_buckets_item+bucket_id_item, dtype=torch.long)
		else:
			users = torch.tensor(self.users[data_index], dtype=torch.long)
			movies = torch.tensor(self.movies[data_index], dtype=torch.long)
		ratings = torch.tensor(self.ratings[data_index], dtype=torch.float32)
		return users, movies, ratings

def calculate_val_loss(args, model, val_dataloader):
	model.eval()
	total_val_loss = 0
	with torch.no_grad():
		for iteration, batch in enumerate(tqdm(val_dataloader)):
			users, movies, ratings = batch
			users = users.to(args.device)
			movies = movies.to(args.device)
			ratings = ratings.to(args.device)
			model_outputs = model.forward(users, movies)
			val_loss = torch.mean((model_outputs-ratings)**2)
			total_val_loss += val_loss.item()

	total_val_loss = total_val_loss / len(val_dataloader)
	return total_val_loss

def print_nnz(args, model):
	nnz_user_T = np.count_nonzero(model.user_T.data.cpu().numpy())
	nnz_item_T = np.count_nonzero(model.item_T.data.cpu().numpy())
	print("size of user T: {} x {} = {}, nnz in user T: {}".format(args.n_users, args.num_user_anchors, args.n_users*args.num_user_anchors, nnz_user_T))
	print("size of item T: {} x {} = {}, nnz in item T: {}".format(args.m_items, args.num_item_anchors, args.m_items*args.num_item_anchors, nnz_item_T))
	print("size of user A: {} x {} = {}".format(args.num_user_anchors, args.latent_dim, args.num_user_anchors*args.latent_dim))
	print("size of item A: {} x {} = {}".format(args.num_item_anchors, args.latent_dim, args.num_item_anchors*args.latent_dim))
	print ('total nnz params:', nnz_user_T+nnz_item_T+args.num_user_anchors*args.latent_dim+args.num_item_anchors*args.latent_dim)
	return

def train(args, train_dataset, val_dataset, model):
	train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=args.batch_size)
	val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=8, batch_size=args.batch_size)

	params = list(model.parameters())
	if args.sparse:
		params.remove(model.user_T)
		params.remove(model.item_T)
		params_opt = [{'params': params},
			      {'params': [model.user_T, model.item_T], 'regularization': (args.lda, 0.0)}]
	else:
		params_opt = params

	# optimizer = Adam(model.parameters(), lr=args.lr, amsgrad=True)
	optimizer = Yogi(params_opt, lr=args.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.num_epoch)

	model.zero_grad()

	best_val_loss = math.inf
	for epoch in range(args.num_epoch):
		model.train()
		total_train_loss = 0
		for iteration, batch in enumerate(tqdm(train_dataloader)):
			users, movies, ratings = batch
			users = users.to(args.device)
			movies = movies.to(args.device)
			ratings = ratings.to(args.device)

			model_outputs = model.forward(users, movies)
			# pdb.set_trace()
			train_loss = torch.mean((model_outputs-ratings)**2)

			train_loss.backward()
			optimizer.step()
			scheduler.step()
			model.zero_grad()
			curr_lr = scheduler.get_lr()[0]

			#if args.sparse:
			#	model.user_T.data = F.relu(model.user_T.data - curr_lr*args.lda)
			#	model.item_T.data = F.relu(model.item_T.data - curr_lr*args.lda)

			total_train_loss += train_loss.item()
			curr_train_loss = total_train_loss/(iteration+1)

			if iteration % 1000 == 0:
				print('curr lr:', curr_lr)
				print("Training loss {}".format(curr_train_loss))
				if args.sparse:
					print_nnz(args, model)
				if args.md:
					print ('embedding_user dim: {}'.format(args.total_user_dim))
					print ('embedding_item dim: {}'.format(args.total_item_dim))
					print ('total params:', args.total_user_dim + args.total_item_dim)

				print (torch.min(model_outputs), torch.min(ratings))
				print (torch.max(model_outputs), torch.max(ratings))
				print (torch.mean(model_outputs), torch.mean(ratings))

		curr_val_loss = calculate_val_loss(args, model, val_dataloader)
		curr_train_loss = total_train_loss/iteration
		print("Training loss {}".format(curr_train_loss))
		print("Val loss {}".format(curr_val_loss))
		if args.sparse:
			print_nnz(args, model)

		if curr_val_loss < best_val_loss:
			best_val_loss = curr_val_loss

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}
			torch.save(checkpoint, args.model_path + 'checkpoint' + str(epoch) + '.pt')

		sys.stdout.flush()


def test(args, test_dataset, model):
	test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=8, batch_size=args.batch_size)

	all_checkpoints = []
	for file in os.listdir(args.model_path):
		if file.endswith(".pt"):
			all_checkpoints.append(os.path.join(args.model_path, file))

	all_checkpoints = sorted(all_checkpoints, key=lambda path: int(path[path.index('point')+5:-3]))
	checkpoint_path = all_checkpoints[-1]
	print ('loading from:', checkpoint_path)
	# pdb.set_trace()
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model'])
	model.to(args.device)
	model.eval()

	if args.sparse:
		print_nnz(args, model)
	elif args.md:
		print ('embedding_user dim: {}'.format(args.total_user_dim))
		print ('embedding_item dim: {}'.format(args.total_item_dim))
		print ('total params:', args.total_user_dim + args.total_item_dim)
	else:
		print ('embedding_user shape: {} x {} = {}'.format(args.n_users, args.latent_dim, args.n_users*args.latent_dim))
		print ('embedding_item shape: {} x {} = {}'.format(args.m_items, args.latent_dim, args.m_items*args.latent_dim))
		print ('total params:', args.n_users*args.latent_dim+args.m_items*args.latent_dim)

	total_loss = 0
	with torch.no_grad():
		for iteration, batch in enumerate(tqdm(test_dataloader)):
			users, movies, ratings = batch
			users = users.to(args.device)
			movies = movies.to(args.device)
			ratings = ratings.to(args.device)
			model_outputs = model.forward(users, movies)
			loss = torch.mean((model_outputs-ratings)**2)
			total_loss += loss.item()

	total_loss = total_loss / len(test_dataloader)
	print("Test loss {}".format(total_loss))
	return total_loss

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path", default='MF', type=str)	# MF 	mdMF 	sparseMF 	NCF 	mdNCF 	sparseNCF
	parser.add_argument("--num_epoch", default=50, type=int)
	parser.add_argument("--batch_size", default=32, type=int)
	parser.add_argument("--optimizer", default='adam', type=str)
	parser.add_argument("--lr", default=1e-2, type=float)
	parser.add_argument("--latent_dim", default=16, type=int)
	parser.add_argument("--num_anchors", default=500, type=int)
	parser.add_argument("--lda", default=1e-5, type=float)
	parser.add_argument("--test", help='testing', action='store_true')

	# for md embeddings
	parser.add_argument("--base_dim", default=16, type=int)
	parser.add_argument("--temperature", default=0.6, type=float)
	parser.add_argument("--k", default=8, type=int)
	parser.add_argument("--round_dims", default=1, type=int)

	args = parser.parse_args()
	cuda = torch.cuda.is_available()
	args.device = torch.device("cuda" if cuda else "cpu")

	args.model_path = 'saved_models/' + args.model_path

	args.sparse = 'sparse' in args.model_path
	args.md = 'md' in args.model_path

	if args.sparse:
		args.model_path += '_anchors%d_lda%0.3f' %(args.num_anchors, args.lda)
	elif args.md:
		args.model_path += '_base%d_temp%0.1f_k%d' %(args.base_dim, args.temperature, args.k)

	args.model_path += '_dim%d' %(args.latent_dim)
	args.model_path += '/'
	if not os.path.exists(args.model_path):
		os.mkdir(args.model_path)

	print (args.model_path)

	args.num_user_anchors = args.num_anchors
	args.num_item_anchors = args.num_anchors

	args.dataset = '25m'
	if args.dataset == '1m':
		RATINGS_CSV_FILE = 'ml1m_ratings.csv'
		ratings = pd.read_csv(RATINGS_CSV_FILE,
							  sep='\t', 
							  encoding='latin-1', 
							  usecols=['userid', 'movieid', 'user_emb_id', 'movie_emb_id', 'rating'])
		user_id_header = 'userid'
		movie_id_header = 'movieid'
	elif args.dataset == '25m':
		RATINGS_CSV_FILE = 'ml-25m/ratings.csv'
		ratings = pd.read_csv(RATINGS_CSV_FILE,
							  sep=',', 
							  encoding='latin-1', 
							  usecols=['userId', 'movieId', 'rating', 'timestamp'])
		user_id_header = 'userId'
		movie_id_header = 'movieId'
	RNG_SEED = 1446557

	args.n_users = ratings[user_id_header].drop_duplicates().max()
	args.m_items = ratings[movie_id_header].drop_duplicates().max()
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

	if args.md:
		# users_train = [1,5,4,4,3,3,6,3,4,4,1]
		# users_val = [1,4,2,3,5,6,0]
		# users_test = [5,5,4,6,7,2]
		# args.n_users = 8
		
		args.md_nums_user, args.md_dims_user, args.total_user_dim, \
		users_train, users_val, users_test \
		= init_md(args, users_train, users_val, users_test, args.n_users)
		args.md_nums_item, args.md_dims_item, args.total_item_dim, \
		movies_train, movies_val, movies_test \
		= init_md(args, movies_train, movies_val, movies_test, args.m_items)

		args.num_buckets_user = len(args.md_nums_user)
		args.num_buckets_item = len(args.md_nums_item)
		# pdb.set_trace()

	train_dataset = MovieDataset(args, users_train, movies_train, ratings_train)
	val_dataset = MovieDataset(args, users_val, movies_val, ratings_val)
	test_dataset = MovieDataset(args, users_test, movies_test, ratings_test)

	if 'NCFModel' in args.model_path:
		args.lr = 1e-2
		model = NCFModel(args)
	elif 'MF' in args.model_path:
		args.lr = 1e-2
		model = MFModel(args)

	model.to(args.device)

	if args.test:
		test(args, test_dataset, model)
	else:
		train(args, train_dataset, val_dataset, model)
	
def init_md(args, data_train, data_val, data_test, total_indices):
	# add all indices (including not in train data), with freq = 1, indices start from 0
	data_train_padded = np.concatenate([np.array(data_train), np.array(range(total_indices))], axis=0)
	freq_counter = Counter(data_train_padded)
	# for other in tqdm():
	# 	if other not in data_train:
	# 		freq_counter[other] = 0
	freq_counter = freq_counter.most_common()
	freqs = torch.tensor(np.array([f for (k, f) in freq_counter]))
	freq_index = {}
	for index, (k, f) in enumerate(freq_counter):
		freq_index[k] = index
	indices_train = torch.tensor(np.array([freq_index[k] for k in data_train]))
	indices_val = torch.tensor(np.array([freq_index[k] for k in data_val]))
	indices_test = torch.tensor(np.array([freq_index[k] for k in data_test]))

	tau = total_indices
	each_block = sum(freqs) / args.k
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
			if bucket < k-1:
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
	nums = torch.tensor(nums)
	ps = nums/tau
	lda = args.base_dim*ps[0]**(args.temperature)
	dims = lda*ps**(-args.temperature)
	for i in tqdm(range(len(dims))):
		if i == 0:
			dims[i] = args.base_dim
		if dims[i] < 1 or torch.isnan(dims[i]):
			dims[i] = 1
	if args.round_dims:
		dims = 2 ** torch.round(torch.log2(dims.type(torch.float)))
	
	args.md_nums = nums
	args.md_dims = dims
	total_dim = sum([a*d for (a,d) in zip(nums, dims)]).item()

	nums = nums.tolist()
	dims = [int(d) for d in dims.tolist()]
	indices_train = [index_to_bucket[k] for k in indices_train.data.numpy()]
	indices_val = [index_to_bucket[k] for k in indices_val.data.numpy()]
	indices_test = [index_to_bucket[k] for k in indices_test.data.numpy()]
	return nums, dims, total_dim, indices_train, indices_val, indices_test

if __name__ == "__main__":
	main()
	
