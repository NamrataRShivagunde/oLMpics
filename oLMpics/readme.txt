# for sequencial partition data
CUDA_VISIBLE_DEVICES=1, python gpt2_mc_mlm.py gpt2 True 

# for random partition data
CUDA_VISIBLE_DEVICES=1, python gpt2_mc_mlm.py gpt2 False 