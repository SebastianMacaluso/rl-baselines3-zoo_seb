import argparse
import os
import sys

import numpy as np
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict, create_test_env, get_model_path, get_saved_hyperparams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument("--video-name", "--video_name", help="Video name", type=str)
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="Number of timesteps to run the environment", default=1000, type=int)
    parser.add_argument("--shift", help="Starting step to record videos", default=1000, type=int)
    parser.add_argument("--custom-video-lenght", "--custom-video-lenght", help="custom_video_lenght", default=100, type=int)
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    parser.add_argument("--n-videos", help="Number of videos to generate", default=1, type=int)
    parser.add_argument("--render_fps", help="Video frame rate", default=30, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )

    parser.add_argument(
        "--video_id",
        type=int,
        help="id identifying each video",
    )
    args = parser.parse_args()

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder
    video_folder = args.output_folder
    seed = args.seed
    video_length = args.n_timesteps
    n_envs = args.n_envs
    # print("custom_video_lenght = ", type(args.custom_video_lenght))

    name_prefix, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    print(f"Loading {model_path}")
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(args.seed)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    # Force rgb_array rendering (gym 0.26+)
    env_kwargs.update(render_mode="rgb_array")

    env = create_test_env(
        env_name.gym_id,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=None,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    print(f"Loading {model_path}")

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    # Deterministic by default except for atari games
    stochastic = args.stochastic or (is_atari or is_minigrid) and not args.deterministic
    deterministic = not stochastic

    if video_folder is None:
        video_folder = os.path.join(log_path, "videos")
    else:
        video_folder = os.path.join(log_path, video_folder)

    # stop = False
    i_video = 0
    i_step = 0 #step at which we start recording a video
    shift = args.shift
    save_clip = False

    # Note: apparently it renders by default
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == i_step, #This means we start rercording from step = i_step. If not change to another step (integer)
        video_length=video_length,
        name_prefix=name_prefix,
        render_fps=args.render_fps,
        video_name=args.video_name,
    )

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)


    try:
        # for _ in range(video_length):
        while i_video < args.n_videos and i_step < video_length:
            i_step+=1



            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            if not args.no_render:
                env.render()
            obs, reward, dones, information = env.step(action)  # type: ignore[assignment]

            if reward > 0:
                if args.video_name == "pad_hits_ball" and i_step >= shift:
                    # For shifts = [0,500], use custom_video_lenght = 80 and cut last 10 frames 
                    custom_video_lenght = 80
                    ##"pad_hits_ball"
                    temp_frames = env.video_recorder.recorded_frames
                    env.video_recorder.recorded_frames = env.video_recorder.recorded_frames[-custom_video_lenght:-5] #keep only the last relevant frames
                    # env.close() # closes environment and saves the video

                    save_clip = True


                if args.video_name == "ball_destroys_bricks" and i_step >= shift:

                    for _ in range(5):
                        i_step+=1
                        action, lstm_states = model.predict(
                            obs,  # type: ignore[arg-type]
                            state=lstm_states,
                            episode_start=episode_starts,
                            deterministic=deterministic,
                        )
                        if not args.no_render:
                            env.render()
                        obs, reward, dones, information = env.step(action)  # type: ignore[assignment]

                    custom_video_lenght = 60
                    ##"pad_hits_ball"
                    temp_frames = env.video_recorder.recorded_frames
                    env.video_recorder.recorded_frames = env.video_recorder.recorded_frames[-custom_video_lenght::] #keep only the last relevant frames
                    # env.close() # closes environment and saves the video

                    save_clip = True


            if dones:

                if args.video_name == "pad_misses_to_hit_ball" and i_step >= shift:

                    # for _ in range(5):
                    #     i_step+=1
                    #     action, lstm_states = model.predict(
                    #         obs,  # type: ignore[arg-type]
                    #         state=lstm_states,
                    #         episode_start=episode_starts,
                    #         deterministic=deterministic,
                    #     )
                    #     if not args.no_render:
                    #         env.render()
                    #     obs, reward, dones, information = env.step(action)  # type: ignore[assignment]

                    custom_video_lenght = 60
                    ##"pad_hits_ball"
                    temp_frames = env.video_recorder.recorded_frames
                    env.video_recorder.recorded_frames = env.video_recorder.recorded_frames[-custom_video_lenght::] #keep only the last relevant frames
                    # env.close() # closes environment and saves the video

                    save_clip = True

            if save_clip:
                            
                env.save_video(str(i_step)+"_"+str(i_video)) #save video without closing environment
                env.video_recorder.recorded_frames = temp_frames
                    
                save_clip = False
                i_video+=1

                # if i_video < args.n_videos:
                #     # Note: apparently it renders by default
                #     env = VecVideoRecorder(
                #         env,
                #         video_folder,
                #         record_video_trigger=lambda x: x == i_step, #This means we start rercording from step 0. If not change to another step (integer)
                #         video_length=video_length,
                #         name_prefix=name_prefix,
                #         render_fps=args.render_fps,
                #         video_name=args.video_name,
                #     )

                #     obs = env.reset()
                #     lstm_states = None
                #     episode_starts = np.ones((env.num_envs,), dtype=bool)

            # if dones:
            #     print("Reward = ", reward, "dones = ", dones, " Information = ", information)
            #     print("------"*10)
            #     stop = True
            #     env.video_recorder.recorded_frames = env.video_recorder.recorded_frames[-args.custom_video_lenght::] #keep only the last relevant frames

            #     env.close() # closes environment and saves the video

            # if reward>0 or information[0]['lives']<5:
            #     print("Reward = ", reward, " Information = ", information)
            #     print("------"*10)
            
            episode_starts = dones
    except KeyboardInterrupt:
        pass

    env.video_recorder.recorded_frames = [] # so that we don't save a final video
    env.close()
    # env.video_recorder.env.close()





    # try:
    #     for _ in range(video_length):
    #         action, lstm_states = model.predict(
    #             obs,  # type: ignore[arg-type]
    #             state=lstm_states,
    #             episode_start=episode_starts,
    #             deterministic=deterministic,
    #         )
    #         if not args.no_render:
    #             env.render()
    #         obs, reward, dones, information = env.step(action)  # type: ignore[assignment]
    #         if dones:
    #             print("Reward = ", reward, "dones = ", dones, " Information = ", information)
    #             print("------"*10)
    #         # if reward>0 or information[0]['lives']<5:
    #         #     print("Reward = ", reward, " Information = ", information)
    #         #     print("------"*10)
            
    #         episode_starts = dones
    # except KeyboardInterrupt:
    #     pass

    # env.close()
