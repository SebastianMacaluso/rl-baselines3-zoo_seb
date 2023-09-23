import os



# env_id = "CartPole-v1"
# env_id = "ALE/SpaceInvaders-v5"

# env_id = "SpaceInvadersNoFrameskip-v4"
env_id = "BreakoutNoFrameskip-v4"


algo = "ppo"

os.system("python train.py --algo "+algo+" --env "+env_id+" --eval-freq 10000 --eval-episodes 10 --n-eval-envs 1")