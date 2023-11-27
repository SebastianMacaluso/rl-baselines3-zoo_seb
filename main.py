import os



# env_id = "CartPole-v1"
# env_id = "ALE/SpaceInvaders-v5"

env_id = "SpaceInvadersNoFrameskip-v4"
# env_id = "BreakoutNoFrameskip-v4"
# env_id = "ALE/Freeway-v5"

algo = "ppo"

save_freq = 500000
# save_freq = int(4e6)

os.system("python train.py --algo "+algo+" --env "+env_id+" --save-freq "+str(save_freq)+"   --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1")




# -------------------------
# CUDA_VISIBLE_DEVICES=1 python main.py