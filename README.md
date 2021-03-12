# Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies

> Pytorch implementation for Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies

Correspondence to: 
  - Paul Liang (pliang@cs.cmu.edu)
  - Irene Li (mengzeli@cs.cmu.edu)

## Paper

[**Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies**](https://arxiv.org/abs/2003.08197)<br>
[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/), Manzil Zaheer, Yuan Wang, Amr Ahmed<br>
ICLR 2021

If you find this repository useful, please cite our paper:
```
@inproceedings{liang2021anchor,
  author    = {Paul Pu Liang and
               Manzil Zaheer and
               Yuan Wang and
               Amr Ahmed},
  title     = {Anchor & Transform: Learning Sparse Embeddings for Large Vocabularies},
  booktitle = {9ths th International Conference on Learning Representations, {ICLR} 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=Vd7lCMvtLqg}
}
```

## Installation

First check that the requirements are satisfied:</br>
Python 3.6</br>
torch 1.2.0</br>
huggingface transformers</br>
numpy 1.18.1</br>
sklearn 0.20.0</br>
matplotlib 3.1.2</br>
gensim 3.8.0 </br>
tqdm 4.45.0</br>
regex 2.5.77</br>
pattern3</br>

## Instructions

download http://files.grouplens.org/datasets/movielens/ml-25m.zip and unzip into a folder ml-25m/

download http://files.grouplens.org/datasets/movielens/ml-1m.zip and unzip into a folder ml-1m/

run python3 movielens_data.py, which extracts the .dat files in ml-1m/ and generates ml1m_ratings.csv

by now, make sure you have ml-25m/ratings.csv and ml1m_ratings.csv

============================================================================================================================

## run full grid search: see generate_grid() in movielens.py

python3 movielens.py --model_path sparseMF --latent_dim 16 --user_anchors 50 --item_anchors 10 --lda2 0.0001 --dataset 1m

## run 25 million

baseline: python3 movielens.py --model_path MF --latent_dim 16 --dataset 25m

mixed dim: python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.4 --k 8 --dataset 25m

ours: python3 movielens.py --model_path sparseMF --latent_dim 16 --user_anchors 50 --item_anchors 10 --lda2 0.0001 --dataset 25m

## run dynamic number of anchors

python3 movielens.py --model_path sparseMF --latent_dim 16 --lda1 0.01 --lda2 0.0001 --dataset 1m --dynamic

============================================================================================================================

## train commands:

python3 movielens.py --model_path MF --latent_dim 16

python3 movielens.py --model_path sparseMF --num_anchors 50 --lda 0.01

python3 movielens.py --model_path sparseMF --num_anchors 100 --lda 0.01

python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.6 --k 8

python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.6 --k 16

python3 movielens.py --model_path NCF --latent_dim 32

python3 movielens.py --model_path sparseNCF --latent_dim 32 --num_anchors 50 --lda 0.01

python3 movielens.py --model_path sparseNCF --latent_dim 32 --num_anchors 100 --lda 0.01

## test commands:

python3 movielens.py --model_path MF --latent_dim 16 --test

python3 movielens.py --model_path sparseMF --num_anchors 50 --lda 0.01 --test

python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.6 --k 8 --test
