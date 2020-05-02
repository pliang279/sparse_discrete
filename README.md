# sparse_discrete

train commands:

python3 movielens.py --model_path MF --latent_dim 16 > res/MF_dim16.txt &

python3 movielens.py --model_path sparseMF --num_anchors 50 --lda 0.01 > res/sparse50_0.01_dim32.txt &
python3 movielens.py --model_path sparseMF --num_anchors 100 --lda 0.01 > res/sparse100_0.01_dim32.txt &

python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.6 --k 8 > res/md16_t0.6_k8.txt
python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.6 --k 16 > res/md16_t0.6_k16.txt

python3 movielens.py --model_path NCF --latent_dim 32 > resNCF/baseline32.txt
python3 movielens.py --model_path sparseNCF --latent_dim 32 --num_anchors 50 --lda 0.01 > resNCF/sparse50_0.01_dim32.txt
python3 movielens.py --model_path sparseNCF --latent_dim 32 --num_anchors 100 --lda 0.01 > resNCF/sparse100_0.01_dim32.txt

test commands:

python3 movielens.py --model_path MF --latent_dim 16 --test

python3 movielens.py --model_path sparseMF --num_anchors 50 --lda 0.01 --test

python3 movielens.py --model_path mdMF --base_dim 16 --temperature 0.6 --k 8 --test
