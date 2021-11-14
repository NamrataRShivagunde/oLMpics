# for sequencial partition data
CUDA_VISIBLE_DEVICES=1, python gpt2_mc_mlm.py gpt2 True 

# for random partition data
CUDA_VISIBLE_DEVICES=1, python gpt2_mc_mlm.py gpt2 False 


QA task

CUDA_VISIBLE_DEVICES=1, python gpt2_mc_qa.py gpt2 cpu
CUDA_VISIBLE_DEVICES=1, python gpt2_mc_qa.py EleutherAI/gpt-neo-2.7B gpu 
CUDA_VISIBLE_DEVICES=1, python gpt2_mc_qa.py EleutherAI/gpt-j-6B gpu 