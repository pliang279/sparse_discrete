# CFModel.py
#
# A simple implementation of matrix factorization for collaborative filtering
# expressed as a Keras Sequential model. This code is based on the approach
# outlined in [Alkahest](http://www.fenris.org/)'s blog post
# [Collaborative Filtering in Keras](http://www.fenris.org/2016/03/07/collaborative-filtering-in-keras).
#
# License: MIT. See the LICENSE file for the copyright notice.
#
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import *
from embeddings import *

def init_weight_old(n, m, scale=1):
	limit = np.sqrt(6 / ((n+m)/scale))
	return np.random.uniform(low=-limit, high=limit, size=(n, m)).astype(np.float32)

def init_weight(n, m, scale=1):
	limit = np.sqrt(2 / n)
	return np.abs(limit*np.random.randn(n, m)).astype(np.float32)

def init_weight_fun(n, m, scale=1):
	limit = np.sqrt(2 / 200000)
	return np.abs(limit*np.random.randn(n, m)).astype(np.float32)

class DynamicMFModel(nn.Module):
	def __init__(self, args):
		super(DynamicMFModel, self).__init__()
		self.args = args
		self.num_users = args.n_users
		self.num_items = args.m_items
		self.user_anchors = args.user_anchors
		self.item_anchors = args.item_anchors
		self.latent_dim = args.latent_dim

		self.max_user_anchors = 100
		self.max_item_anchors = 100

		scale = self.max_user_anchors / self.user_anchors

		self.all_user_T = torch.nn.Parameter(torch.tensor(init_weight(self.num_users, self.max_user_anchors, scale=scale)))
		self.all_user_A = torch.nn.Parameter(torch.tensor(init_weight(self.max_user_anchors, self.latent_dim, scale=scale)))
		self.all_item_T = torch.nn.Parameter(torch.tensor(init_weight(self.num_items, self.max_item_anchors, scale=scale)))
		self.all_item_A = torch.nn.Parameter(torch.tensor(init_weight(self.max_item_anchors, self.latent_dim, scale=scale)))

	def expand_user(self):
		if self.user_anchors <= self.max_user_anchors - self.args.delta:
			self.user_anchors += self.args.delta

	def expand_item(self):
		if self.item_anchors <= self.max_item_anchors - self.args.delta:
			self.item_anchors += self.args.delta

	def reduce_user(self):
		if self.user_anchors >= self.args.delta:
			self.user_anchors -= self.args.delta

	def reduce_item(self):
		if self.item_anchors >= self.args.delta:
			self.item_anchors -= self.args.delta

	def get_nnz(self):
		self.user_T = self.all_user_T[:,:self.user_anchors]
		self.item_T = self.all_item_T[:,:self.item_anchors]
		nnz_user_T = np.count_nonzero(self.user_T.data.cpu().numpy())
		nnz_item_T = np.count_nonzero(self.item_T.data.cpu().numpy())
		return nnz_user_T, nnz_item_T
	
	def forward(self, user_indices, item_indices):
		self.user_T = self.all_user_T[:,:self.user_anchors]
		self.user_A = self.all_user_A[:self.user_anchors]
		self.item_T = self.all_item_T[:,:self.item_anchors]
		self.item_A = self.all_item_A[:self.item_anchors]
		# print ('user T', self.user_T.min(), self.user_T.max())
		# print ('user A', self.user_A.min(), self.user_A.max())
		user_embedding = torch.matmul(self.user_T[user_indices], self.user_A)
		item_embedding = torch.matmul(self.item_T[item_indices], self.item_A)
		# print ('user_embedding', user_embedding.min(), user_embedding.max())
		# print ('item_embedding', item_embedding.min(), item_embedding.max())
		rating = torch.einsum('ni,ni->n', user_embedding, item_embedding)
		return rating

class MFModel(nn.Module):
	def __init__(self, args):
		super(MFModel, self).__init__()
		self.args = args
		self.num_users = args.n_users
		self.num_items = args.m_items
		self.user_anchors = args.user_anchors
		self.item_anchors = args.item_anchors
		self.latent_dim = args.latent_dim
		
		self.bias = torch.nn.Parameter(torch.tensor(0.0))

		if args.sparse:
			self.user_T = torch.nn.Parameter(torch.tensor(init_weight(self.num_users, self.user_anchors)))
			self.item_T = torch.nn.Parameter(torch.tensor(init_weight(self.num_items, self.item_anchors)))
			self.user_A = torch.nn.Parameter(torch.tensor(init_weight(self.user_anchors, self.latent_dim)))
			self.item_A = torch.nn.Parameter(torch.tensor(init_weight(self.item_anchors, self.latent_dim)))
		elif args.md:
			self.embeddings_user, self.projs_user = self.create_md_emb(args.md_nums_user, args.md_dims_user)
			self.embeddings_item, self.projs_item = self.create_md_emb(args.md_nums_item, args.md_dims_item)
		else:
			# self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
			# self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
			self.embedding_user = torch.nn.Parameter(torch.tensor(init_weight(self.num_users, self.latent_dim)))
			self.embedding_item = torch.nn.Parameter(torch.tensor(init_weight(self.num_items, self.latent_dim)))

	def create_md_emb(self, nums, dims):
		embeddings = nn.ParameterList([])
		for (num, dim) in zip(nums, dims):
			emb = torch.nn.Parameter(torch.tensor(init_weight(num, dim)))
			embeddings.append(emb)
		projs = nn.ParameterList([])
		for dim in dims:
			if dim == self.args.latent_dim:  # this is just identity
				proj = torch.nn.Parameter(torch.eye(self.args.latent_dim), requires_grad=False)
			else:
				proj = torch.nn.Parameter(torch.tensor(init_weight(dim, self.args.base_dim)))
			projs.append(proj)
		return embeddings, projs

	def apply_md_emb(self, indices, embeddings, projs):
		full_embedding = []
		for (embedding, proj) in zip(embeddings, projs):
			small_emb = torch.matmul(embedding, proj)
			full_embedding.append(small_emb)
		full_embedding = torch.cat(full_embedding, dim=0)
		return full_embedding[indices]

	def get_nnz(self):
		nnz_user_T = np.count_nonzero(self.user_T.data.cpu().numpy())
		nnz_item_T = np.count_nonzero(self.item_T.data.cpu().numpy())
		return nnz_user_T, nnz_item_T

	def forward(self, user_indices, item_indices):
		if self.args.sparse:
			user_embedding = torch.matmul(self.user_T[user_indices], self.user_A)
			item_embedding = torch.matmul(self.item_T[item_indices], self.item_A)
		elif self.args.md:
			user_embedding = self.apply_md_emb(user_indices, self.embeddings_user, self.projs_user)
			item_embedding = self.apply_md_emb(item_indices, self.embeddings_item, self.projs_item)
		else:
			user_embedding = self.embedding_user[user_indices]	# self.embedding_user(user_indices)
			item_embedding = self.embedding_item[item_indices]      # self.embedding_item(item_indices)
		rating = torch.einsum('ni,ni->n', user_embedding, item_embedding) + self.bias
		return rating

class NCFModel(nn.Module):
	def __init__(self, args):
		super(NCFModel, self).__init__()
		self.args = args
		self.num_users = args.n_users
		self.num_items = args.m_items
		self.user_anchors = args.user_anchors
		self.item_anchors = args.item_anchors
		self.latent_dim = args.latent_dim

		if args.sparse:
			self.user_T = torch.nn.Parameter(torch.tensor(init_weight(self.num_users, self.user_anchors)))
			self.item_T = torch.nn.Parameter(torch.tensor(init_weight(self.num_items, self.item_anchors)))
			self.user_A = torch.nn.Parameter(torch.tensor(init_weight(self.user_anchors, self.latent_dim)))
			self.item_A = torch.nn.Parameter(torch.tensor(init_weight(self.item_anchors, self.latent_dim)))
		elif args.md:
			base_dim = self.latent_dim
			self.embedding_user = PrEmbeddingBag(self.num_users, self.latent_dim, base_dim)
			self.embedding_item = PrEmbeddingBag(self.num_items, self.latent_dim, base_dim)
		else:
			self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
			self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

		self.layers = [self.latent_dim*2]
		self.fc_layers = torch.nn.ModuleList()
		for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
			self.fc_layers.append(torch.nn.Linear(in_size, out_size))

		self.drop_layer = torch.nn.Dropout(p=0.5)
		self.affine_output = torch.nn.Linear(in_features=self.layers[-1], out_features=1)
		self.logistic = torch.nn.Sigmoid()

	def forward(self, user_indices, item_indices):
		if self.args.sparse:
			user_embedding = torch.matmul(self.user_T[user_indices], self.user_A)
			item_embedding = torch.matmul(self.item_T[item_indices], self.item_A)
		else:
			user_embedding = self.embedding_user(user_indices)
			item_embedding = self.embedding_item(item_indices)
		vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector

		for idx, _ in enumerate(range(len(self.fc_layers))):
			vector = self.fc_layers[idx](vector)
			vector = self.drop_layer(vector)
			vector = torch.nn.ReLU()(vector)

		logits = self.affine_output(vector)
		# rating = self.logistic(logits)
		rating = logits
		return rating


if __name__ == "__main__":
	pass
