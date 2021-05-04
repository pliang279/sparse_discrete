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


def load_data():
	RATINGS_CSV_FILE = 'amazon_data/all_csv_files.csv'
	ratings = pd.read_csv(RATINGS_CSV_FILE,
						  sep=',',
						  encoding='latin-1',
						  usecols=[0,1,2,3],
						  names=['userID', 'itemID', 'rating', 'time'],
						  header=None)
	user_id_header = 'userID'
	item_id_header = 'itemID'
	print (len(ratings), 'ratings loaded.')

	RNG_SEED = 1446557
	users = ratings[user_id_header]# .drop_duplicates()
	items = ratings[item_id_header]# .drop_duplicates()

	def filter_by_freq(df: pd.DataFrame, column: str, min_freq: int) -> pd.DataFrame:
		"""Filters the DataFrame based on the value frequency in the specified column.

		:param df: DataFrame to be filtered.
		:param column: Column name that should be frequency filtered.
		:param min_freq: Minimal value frequency for the row to be accepted.
		:return: Frequency filtered DataFrame.
		"""
		# Frequencies of each value in the column.
		freq = df[column].value_counts()
		# Select frequent values. Value is in the index.
		frequent_values = freq[freq >= min_freq].index
		# Return only rows with value frequency above threshold.
		return df[df[column].isin(frequent_values)]

	prev_num = len(ratings)
	print (prev_num)
	while True:
		ratings = filter_by_freq(ratings, user_id_header, 5)
		ratings = filter_by_freq(ratings, item_id_header, 5)
		new_num = len(ratings)
		print (new_num)
		if new_num == prev_num:
			break
		else:
			prev_num = new_num

	# pdb.set_trace()

	def learn_map(raw_items):
		forward_map = dict()
		rev_map = dict()
		mapped_items = []
		num_items = 0
		for item in tqdm(raw_items):
			if item not in forward_map:
				forward_map[item] = num_items
				rev_map[num_items] = item
				num_items += 1
			mapped_items.append(forward_map[item])
		mapped_items = np.array(mapped_items)
		return mapped_items, num_items, forward_map, rev_map

	shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
	ratings = shuffled_ratings['rating'].values
	raw_users = shuffled_ratings[user_id_header].values
	raw_items = shuffled_ratings[item_id_header].values
	mapped_users, num_users, user_forward_map, user_rev_map = learn_map(raw_users)
	mapped_items, num_items, item_forward_map, item_rev_map = learn_map(raw_items)

	n_users = num_users
	print ('Users:', mapped_users, ', shape =', mapped_users.shape, 'num =', n_users)
	m_items = num_items
	print ('Items:', mapped_items, ', shape =', mapped_items.shape, 'num =', m_items)
	print ('Ratings:', ratings, ', shape =', ratings.shape)
	# pdb.set_trace()

	hf = h5py.File('amazon_data/saved_amazon_data_filtered5.h5', 'w')
	hf.create_dataset('users', data=mapped_users)
	hf.create_dataset('items', data=mapped_items)
	hf.create_dataset('ratings', data=ratings)
	hf.close()

	pickle.dump(user_forward_map, open("amazon_data/user_forward_map_filtered5.pkl", "wb"))
	pickle.dump(user_rev_map, open("amazon_data/user_rev_map_filtered5.pkl", "wb"))
	pickle.dump(item_forward_map, open("amazon_data/item_forward_map_filtered5.pkl", "wb"))
	pickle.dump(item_rev_map, open("amazon_data/item_rev_map_filtered5.pkl", "wb"))
	print ('data loaded and saved')
	pdb.set_trace()


if __name__ == "__main__":
	load_data()


