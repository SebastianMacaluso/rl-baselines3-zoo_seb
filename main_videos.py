import os



# env_id = "CartPole-v1"
# env_id = "ALE/SpaceInvaders-v5"

# env_id = "SpaceInvadersNoFrameskip-v4"
# env_id = "BreakoutNoFrameskip-v4"
env_id = "ALE/Freeway-v5"

algo = "ppo"

steps = 1
# Use the best saved model
for i in range(steps):
    os.system("python -m rl_zoo3.record_video --algo "+algo+" --env "+env_id+" -f logs --exp-id 1 -n 500 --load-best --video_id "+str(i))


# Record a video of a checkpoint saved during training (here the checkpoint name is rl_model_10000_steps.zip):
# os.system("python -m rl_zoo3.record_video --algo "+algo+" --env "+env_id+" -n 1000 --load-checkpoint 10000")



# -------------------------
# CUDA_VISIBLE_DEVICES=1 python main_videos.py