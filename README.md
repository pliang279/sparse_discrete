# sparse_discrete

download http://files.grouplens.org/datasets/movielens/ml-25m.zip and unzip into a folder ml-25m/

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
