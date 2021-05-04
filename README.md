# Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies

> Pytorch implementation for Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies

Correspondence to: 
  - Paul Liang (pliang@cs.cmu.edu)
  - Manzil Zaheer (manzilzaheer@google.com)

## Paper

[**Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies**](https://arxiv.org/abs/2003.08197)<br>
[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/), [Manzil Zaheer](http://www.manzil.ml/), [Yuan Wang](https://ai.google/research/people/YuanWang), [Amr Ahmed](https://ai.google/research/people/AmrAhmed)<br>
ICLR 2021

If you find this repository useful, please cite our paper:
```
@inproceedings{liang2021anchor,
  author    = {Paul Pu Liang and
               Manzil Zaheer and
               Yuan Wang and
               Amr Ahmed},
  title     = {Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=Vd7lCMvtLqg}
}
```

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
numpy 1.18.1</br>
matplotlib 3.1.2</br>
tqdm 4.45.0</br>

The next step is to clone the repository:
```bash
git clone https://github.com/pliang279/sparse_discrete.git
```

## Data

### Movielens data

download Movielens 25m data from http://files.grouplens.org/datasets/movielens/ml-25m.zip and unzip into a folder ml-25m/

download Movielens 1m data from http://files.grouplens.org/datasets/movielens/ml-1m.zip and unzip into a folder ml-1m/

run ```python3 movielens_data.py``` which extracts the .dat files in ml-1m/ and generates ml-1m/ml1m_ratings.csv

by now, make sure you have the files ```ml-25m/ratings.csv``` and ```ml-1m/ml1m_ratings.csv```

### Amazon review data

download amazon data from http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/all_csv_files.csv into a folder called amazon_data/

run ```python3 movielens_data.py```, which parses the .csv files in amazon_data/ and generates the file ```amazon_data/saved_amazon_data_filtered5.h5```

## Instructions

### Movielens data

MF baseline: ```python3 movielens.py --model_path MF --latent_dim 16 --dataset 25m```

MixDim embeddings: ```python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.4 --k 8 --dataset 25m```

ANT: ```python3 movielens.py --model_path sparseMF --latent_dim 16 --user_anchors 50 --item_anchors 10 --lda2 0.0001 --dataset 25m```

NBANT: ```python3 movielens.py --model_path sparseMF --latent_dim 16 --lda1 0.01 --lda2 0.0001 --dataset 25m --dynamic```

### Amazon review data


