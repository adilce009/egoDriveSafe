import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from environment.configs import load_config_data
#from l5kit.environment.callbacks_vec import L5KitEvalCallback
#from l5kit.environment.envs.l5_pred_vec import SimulationConfigGym
from environment.env.l5_env import SimulationConfigGym
from environment.feature_extractor import CustomExtractor


os.environ["L5KIT_DATA_FOLDER"] = "/Users/adil/Documents/Research/Dataset/Lyft"
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# get environment configs
env_config_path = '/Users/adil/Documents/Research/Projects/egoDriveSafe/config.yaml'
cfg = load_config_data(env_config_path)
print(cfg)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
train_eps_length = 31 # simulation length will be 30
train_envs = 1

# Evaluate on entire scene (~248 time steps)
eval_eps_length = 30
eval_envs = 1       #4
features_dim = 256

########custom feature extractor backbone
policy_kwargs = {
    "features_extractor_class": CustomExtractor,
    "features_extractor_kwargs": {"features_dim": features_dim},
    "normalize_images": False
}
#"features_extractor_kwargs": {"features_dim": features_dim}, --> this was inside policy_kwarg
########################### make train env
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1

# env configuration
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg,
              'train': True,
              }

# three ways to start a process: spawn, fork (supports in unix system only), and forkserver (should not have a conflict)
#env = make_vec_env("L5-pred-v0", env_kwargs=env_kwargs, n_envs=train_envs,
 #                  vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"}) # makes n_env number of environments for speedingup
env = make_vec_env("ego_SafeDrive-v0", env_kwargs=env_kwargs, n_envs=train_envs,
                   vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"}) # makes n_env number of environments for speedingup

###### # code for observation wrapper
'''
individual_envs = [gym.make('L5-pred-vec-v2', **env_kwargs) for _ in range(train_envs)]
wrapped_envs = [CustomObservationWrapper(env) for env in individual_envs]
vec_env = SubprocVecEnv([lambda: env for env in wrapped_envs], start_method="fork")
'''
# ########################### make eval env
validation_sim_cfg = SimulationConfigGym()
validation_sim_cfg.num_simulation_steps = train_eps_length + 1
eval_env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, \
                   'return_info': True, 'train': False, 'sim_cfg': validation_sim_cfg}
eval_env = make_vec_env("ego_SafeDrive-v0", env_kwargs=eval_env_kwargs, n_envs=eval_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

#features_dim = 123 # for feature extractor

# Custom Feature Extractor backbone. Not using for now
# policy_kwargs = {
#     "features_extractor_class": CustomCombinedExtractor
# }

################## Clipping schedule of PPO epsilon parameter
start_val = 0.1
end_val = 0.01
training_progress_ratio = 1.0
clip_schedule = get_linear_fn(start_val, end_val, training_progress_ratio) # a linear function

################# tensorboard callback
class TenbsorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) ->bool:
        #for i in range(train_envs):
        dist_reward = self.locals["rewards"][1]
        speed_reward = self.locals["rewards"][2]

        #dist_reward = self.locals['self'].env.reward["dist_nbr"]
        #speed_reward = self.locals['self'].env.reward["speed"]

        self.logger.record('distance_reward', dist_reward)
        self.logger.record('speed_reward', speed_reward)
        return True
# ################ Hyperparameters for PPO.
lr = 3e-4
num_rollout_steps = 16# episode length 26
gamma = 0.95
gae_lambda = 0.95
n_epochs = 10   # number of epochs to train ppo
seed = 42
batch_size = 4
ent_coef = 0.001
tensorboard_log = 'tb_log'

# ################## Define the PPO Policy.
#policy_kwargs = dict(net_arch=dict(pi=[32,32], vg=[32,32]))
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=num_rollout_steps,
             learning_rate=lr, gamma=gamma, ent_coef=ent_coef, tensorboard_log=tensorboard_log, n_epochs=n_epochs,
             clip_range=clip_schedule, batch_size=batch_size, seed=seed, gae_lambda=gae_lambda, device = device)
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=num_rollout_steps,
#             learning_rate=lr, gamma=gamma, tensorboard_log=tensorboard_log,
#             clip_range=clip_schedule, batch_size=batch_size, seed=seed, gae_lambda=gae_lambda)
#model = model.to(device)
#3rd arg: policy_kwargs=policy_kwargs,

# ################# Defining Callbacks
#
# We can additionally define callbacks to save model checkpoints and evaluate models during training.
#print(model.policy)
#callback_list = []

################## Save Model Periodically#
#save_freq = 100
save_path = './logs/egoInput-01_embedding_layer_features_ent-0.005_gamma-0.95'
output = 'PPO'

# checkpoint_callback = CheckpointCallback(save_freq=(save_freq // train_envs), save_path=save_path, \
#                                          name_prefix=output)
# callback_list.append(checkpoint_callback)
#
# ################## Eval Model Periodically
# eval_freq = 100
# n_eval_episodes = 1
# val_eval_callback = L5KitEvalCallback(eval_env, eval_freq=(eval_freq // train_envs), \
#                                       n_eval_episodes=n_eval_episodes, n_eval_envs=eval_envs)
# callback_list.append(val_eval_callback)
#
# callback = SaveOnBestTrainingRewardCallback (check_freq=1000, log_dir=save_path)

###################  Train
#
# n_steps = 4
# #model.learn(n_steps, callback=callback_list)
# #model.learn(n_steps = 1000, callback = callback, tb_log_name ='ppo_0.0003')
# for timestep in range(1, n_steps + 1):
#     print('-------- Start Training------------')
#     model.learn(1, callback=callback_list)
#     print('--------End Training----------')
#     print('_________________________________________')

TIMESTEPS = 5000

for i in range(1,2000):

    #model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, callback=TenbsorboardCallback(),tb_log_name = "PPO-14")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="egoInput-01")
    model.save(f"{save_path}/{TIMESTEPS*i}")
    #print('number of training loops finished:', i)

########### Create a TensorBoard writer


# Function to evaluate the agent
def evaluate_model(model, eval_env, num_episodes=10):
    total_reward = 0.0

    for _ in range(num_episodes):
        obs = eval_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward

    mean_reward = total_reward / num_episodes
    return mean_reward

'''
writer = SummaryWriter()

# Train the agent
total_timesteps = 3
log_interval = 1

for timestep in range(1, total_timesteps + 1):
    model.learn(1)

    # Log the training progress
    if timestep % log_interval == 0:
        mean_reward = evaluate_model(model, eval_env)
        writer.add_scalar("Mean Reward", mean_reward, timestep)

# Close the TensorBoard writer
writer.close()

# Save the trained model
model.save("ppo_pred_vec")

'''
