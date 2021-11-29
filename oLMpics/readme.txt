For MC-MLM tasks run

        python gpt2_mc_mlm.py [Modelname] [True/False]

        e.g. CUDA_VISIBLE_DEVICES=1, python gpt2_mc_mlm.py gpt2 False 

        # for age group
        e.g. CUDA_VISIBLE_DEVICES=1, python gpt2_mc_mlm.py gpt2 True 


For MC-QA tasks run

        python gpt2_mc_qa.py [Modelname]

        e.g. CUDA_VISIBLE_DEVICES=1, python gpt2_mc_qa.py gpt2 


Modelname can be {gpt2, gpt2-medium, gpt2-large, gpt2-xl, EleutherAI/gpt-neo-1.3B}