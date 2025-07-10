#towards finding a new evaluation metric that matches the aimed policy

# find distance to the vehicles. first, find distance to the vehicle in front
# there may not be any vehicle in front. how to measure the metric in that case?
# if no car, measure speed;
    # -- normal condition
        # -- no stopping or slowing down condition
    # -- has to stop condition (distance to stop condition, ideal breaking distance)
        # stop conditions: red light, stop sign, crosswalk, other agent
    # -- has to slow down condition
        # -- is it a turn lane?
        # -- any slow or stopped agent?
# if car: measure distance

#imports
import torch

from l5kit.data.map_api import MapAPI
from environment.env.l5_pred_vec import SimulationConfigGym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from l5kit.geometry import transform_points
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.data import LocalDataManager
from l5kit.data import MapAPI
from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene_modified
from bokeh.io import show
from l5kit.visualization.visualizer.visualizer import visualize_modified
from environment.configs import load_config_data
#
# class evaluation():
#     def __init__(self, num_episodes, eval_env):
#         self.MapAPI = MapAPI    # consider getting it from environment, like reward function
#         self.num_episodes = num_episodes
#         self.env = eval_env
#
#     def distance(self):
#         pass
#
#     def speed(self):
#         pass

def eval(eval_env,num_episodes):
# evaluation should reflect the follwoing:
# is it maintaining a good distance?
# is it maintaining a good speed limit?
# is it maintaining the expected lane?
    store_mean_dist= []
    store_mean_speed_err = []
    store_mean_good_speed = []
    store_mean_y_dist_error = []
    env = eval_env

    for ep in range(num_episodes):
        count_episode_length = 0
        mean_eps_distance = 0
        mean_good_speed = 0
        total_distance = 0
        episode_speed_error = 0
        episode_good_speed = 0
        mean_episode_speed_error = 0
        mean_episode_good_speed = 0
        episode_y_displacement_error = 0
        mean_y_disp_error = 0
        obs = env.reset()
        done = False
        while not done:
            action, state = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_distance = total_distance + info[0]["Distance_to_neighbor"]
            episode_speed_error = episode_speed_error + info[0]["speed_error"]
            episode_good_speed = episode_good_speed + info[0]["good_speed"]
            #episode_y_displacement_error = episode_y_displacement_error + info[0]["latitudinal_error"]
            count_episode_length +=1
        if done:
            mean_eps_distance = total_distance / count_episode_length
            mean_episode_speed_error = episode_speed_error/ count_episode_length
            mean_episode_good_speed = episode_good_speed
            #mean_y_disp_error = episode_y_displacement_error/ count_episode_length
            recorded_states = info[0]["recorded_states"]
            simulated_states = info[0]["simulated_states"]
            recorded_ta_poly = recorded_states[2:,:2]
            simulated_ta_poly = simulated_states[2:,:2]
            sim_out = info[0]["sim_out"]
            ta_ins_outs = info[0]["ta_in_out"]
            ta_track_id = info[0]["track_id"]

            ###
            #draw trajectory
            mapAPI = MapAPI.from_cfg(dm, cfg)
            vis_in = simulation_out_to_visualizer_scene_modified(sim_out, mapAPI, ta_ins_outs, ta_track_id)
            show(visualize_modified(0, vis_in)) # we have only one scene in simulation dataset; show(visualize(sim_out.scene_id, vis_in))

            # target_positions_pixels = transform_points(ta_features["target_positions"], ta_features["raster_from_agent"])
            #
            # draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=ta_features["target_yaws"])
            # plt.imshow(im)
            # plt.show()

        store_mean_dist.append(mean_eps_distance)
        store_mean_speed_err.append(mean_episode_speed_error)
        store_mean_good_speed.append(mean_episode_good_speed)
        store_mean_y_dist_error.append(mean_y_disp_error)
        #print('number of episodes elapsed', len(store_mean_dist))
    #print('mean_distances:', store_mean_dist)
    mean_distance = sum(store_mean_dist)/num_episodes
    mean_speed_error = sum(store_mean_speed_err)/num_episodes
    mean_good_speed = sum(store_mean_good_speed)/num_episodes
    mean_y_disp_error = sum(store_mean_y_dist_error)/num_episodes
    #plt.plot()
    return mean_distance, mean_speed_error, mean_good_speed

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
env_config_path = '/Users/adil/Documents/Research/Projects/DriveSafe/PredWithVec/config_vec_pred-2.yaml'
cfg = load_config_data(env_config_path)
train_eps_length = 29
dm = LocalDataManager(None)
validation_sim_cfg = SimulationConfigGym()
validation_sim_cfg.num_simulation_steps = train_eps_length + 1
eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, \
                   'return_info': True, 'train': False, 'sim_cfg': validation_sim_cfg, 'filtered_dataset' : True}
eval_env = make_vec_env("L5-pred-vec-v2", env_kwargs=eval_env_kwargs, n_envs=1,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

#model = PPO.load("logs/PPO-10_reward-no_disp_err_ent-0.001_gamma-0.9/1040000.zip")
model = PPO.load("logs/PPO-05_embedding_layer_features_ent-0.005_gamma-0.9/6745000.zip")
model.device = device
model.policy.action_net.to(device)
model.policy.value_net.to(device)
model.policy.mlp_extractor.policy_net.to(device)
model.policy.mlp_extractor.value_net.to(device)
num_episode = 100
mean_distance, mean_speed_error, mean_good_speed = eval(eval_env, num_episode)
print('mean distance:', mean_distance, 'mean speed error:', mean_speed_error, 'mean_good_speed:', mean_good_speed)
#print('mean_distance', mean_distance)
#print('mean speed error', mean_speed_error)
#print('mean_y_disp_error', mean_y_disp_error)
