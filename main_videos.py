import os



# env_id = "CartPole-v1"
# env_id = "ALE/SpaceInvaders-v5"

# env_id = "SpaceInvadersNoFrameskip-v4"
env_id = "BreakoutNoFrameskip-v4"
# env_id = "ALE/Freeway-v5"

algo = "ppo"
render_fps = 10
load_checkpoint = 1000000
n_steps = 4000  # We currently repeat each action 3 times => For each step we have 3 frames. Total frames f = time * frame_rate and n_steps = time * frame_rate / repeated_actions. So for  render_fps = 25 and time=3sec we have n_steps = 25
custom_video_lenght = 90
video_folder = "videos/success"

video_name = "pad_hits_ball"
# video_name = "pad_moves_towards_ball"
# video_name = "ball_destroys_bricks"
# video_name = "pad_misses_to_hit_ball"
# video_name = "pad_moves_away_from_incoming_ball"

n_videos = 1

steps = 1
# Use the best saved model
for i in range(steps):
    # os.system("python -m rl_zoo3.record_video --render_fps "+str(render_fps)+" --load-checkpoint "+str(load_checkpoint)+" --algo "+algo+" --seed 47 --env "+env_id+" -f logs --exp-id 3 -n "+str(n_steps)+" --video_id "+str(i))

    #load best
    os.system("python -m rl_zoo3.record_video --render_fps "+str(render_fps)+" -o "+str(video_folder)+" -n_videos "+(n_videos)+" --algo "+algo+" --seed 47 --env "+env_id+" -f logs --exp-id 3 -n "+str(n_steps)+" --custom_video_lenght "+str(custom_video_lenght)+" --load-best --video_id "+str(i))

# Record a video of a checkpoint saved during training (here the checkpoint name is rl_model_10000_steps.zip):
# os.system("python -m rl_zoo3.record_video --algo "+algo+" --env "+env_id+" -n 1000 --load-checkpoint 10000")



# -------------------------
# CUDA_VISIBLE_DEVICES=1 python main_videos.py