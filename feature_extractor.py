import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.local_graph import SinusoidalPositionalEmbedding, LocalSubGraph
from environment.global_graph import MultiheadAttentionGlobalHead, VectorizedEmbedding
from environment import models, obs_prep
from common import pad_avail, pad_points, transform_points

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor from raster images for the RL Policy.

    :param observation_space: the input observation space
    :param features_dim: the number of features to extract from the input
    :param model_arch: the model architecture used to extract the features
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 model_arch: str = "simple_gn"):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        num_input_channels = observation_space["image"].shape[0]

        if model_arch == 'simple_gn':
            # A simplified feature extractor with GroupNorm.
            model = models.SimpleCNN_GN(num_input_channels, features_dim)
        else:
            raise NotImplementedError

        extractors = {"image": model}
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = features_dim

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space:gym.spaces.Dict):
        super().__init__(observation_space, features_dim = 64)

        extractors = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == 'tl_status':
                input_size = 5
                #subspace = np.ravel(subspace)
            else:
                input_size =  subspace.high.size
            extractors[key] = nn. Linear(input_size, 16)
            total_concat_size +=16

        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self,observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
            input = observations[key]
            input = torch.flatten(input, 1, -1)
            encoded_tensor_list.append(extractor(input))
        return torch.cat(encoded_tensor_list, dim=1)

class CustomExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space:gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        #model = nn. Linear(256, features_dim=256)


        self._features_dim = features_dim
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        #self.device = torch.device("cpu")
        ###
        self._d_local = 256
        self._d_global = 256
        self._subgraph_layers = 3
        # self.register_buffer("weights_scaling", torch.as_tensor(weights_scaling))
        # self.criterion = criterion
        self._agent_features = ["velocity_x", "velocity_y", "yaw"]
        self._lane_features = ["start_x", "start_y", "tl_feature"]
        self._vector_agent_length = len(self._agent_features)
        self._vector_lane_length = len(self._lane_features)

        self.weights_scaling = [1.0, 1.0]
        self.normalize_targets = True
        self._num_targets = 2  # two outputs to predict
        self._global_head_dropout = 0.0  # from cfg
        num_outputs = len(self.weights_scaling)
        num_timesteps = self._num_targets // num_outputs

        self.input_embed = nn.Linear(self._vector_agent_length, self._d_local)
        #self.input_embed = self.input_embed.to(self.device)
        self.disable_lane_boundaries = True
        self.positional_embedding = SinusoidalPositionalEmbedding(self._d_local)
        self.type_embedding = VectorizedEmbedding(self._d_local)
        #self.positional_embedding = self.positional_embedding.to(self.device)
        self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)
        #self.local_subgraph = self.local_subgraph.to(self.device)
        self.input_dim = 81*256
        self.fc = nn.Linear(self.input_dim, self._d_global)
        self.global_head = MultiheadAttentionGlobalHead(
            self._d_global, num_timesteps, num_outputs, dropout=self._global_head_dropout
        )
        #self.fc = self.fc.to(self.device)
        #extractors = {"linear": self.input_embed}
        #self.extractors = nn.ModuleDict(extractors)

    def check_device(self):
        return self.input_embed.device.type

    def forward(self, observation):
        #preprocess observation
        #observation = observation.to(self.device)
        for key, tensor in observation.items():
            observation[key] = tensor.to(self.device)
        self.to(self.device)
        all_polys, all_avail = self.next_obs_preprocessing(observation)
        #all_polys = all_polys

        #all_polys = all_polys.float().to(self.device)

        all_avail = all_avail.bool()

        #prep_obs_dict = {"embed_feat": prep_obs}
        # for key, extractor in self.extractors.items():
        #     embed_obs = extractor(all_polys)
        polys = self.input_embed(all_polys)
        type_embedding = self.type_embedding(observation).transpose(0, 1)
        pos_embedding = self.positional_embedding(all_polys).unsqueeze(0).transpose(1, 2)
        invalid_mask = ~all_avail
        #invalid_mask = invalid_mask.to(self.device)
        invalid_polys = invalid_mask.all(-1)
        polys = self.local_subgraph(polys, invalid_mask, pos_embedding)
        all_embs = F.normalize(polys, dim=-1) * (self._d_global ** 0.5)
        all_embs = all_embs.transpose(0, 1)  # shape chagne (6,81,256) --> (81,6,256)
        lane_bdry_len = observation['lanes'].shape[1]
        if self.disable_lane_boundaries:
            type_embedding = type_embedding[:-lane_bdry_len]
        invalid_polys[:, 0] = 0  # make AoI always available in global graph
        outputs, attns = self.global_head(all_embs, type_embedding, invalid_polys) # output is a vector of dim 256

        #polys = self.fc(polys.view(len(polys), -1))

        return outputs

    def next_obs_preprocessing(self, obs):
        '''
        this function will take an input observation and will do the following:
        1. prepare polylines for the agents and the other road features
        2. type embedding
        3. code to generate window sized steps
        4. pad agents to match with that of the other features, e.g. lanes, crosswalks, of the road
        5. transform points (need to check it)
        6. standardization
        7. embed inputs
        8. normalization
        '''
        #print(obs)
        # ([mean vel_x, mean val_y], [std_vel_x, std_vel_y], mean_yaw, std_y). These values are only for car agents
        # the stats were calculated from the EgoAgentDatasetVectorized dataset after it was filtered for car agents only
        # stats takes from small dataset. do it for full dataset
        # stats = [-0.0077168634,0.012839081, 1.2878988, 1.3605937, 0.54369116, 1.2016048]# obtained from full dataser
        #stats = [2.5696368, 2.768638,3.6705377, 4.0184836, 0.54369116, 1.2016048]    # mean(abs(non zero vel_x)), std(abs(non zero vel_x))
        stats = [2.5696368, 2.768638, 15.5, 4.0184836, 0.54369116, 1.2016048 ]
        stats = torch.tensor(stats).to(self.device)
        # past and static info

        ego_features = obs

        ego_features["agent_feat"] = torch.unsqueeze(ego_features["agent_feat"], dim=1)
        #print('ta_features:',ta_features)
        agents_past_polys = torch.cat(
        (ego_features["agent_feat"],
          ego_features["other_agents_feat"]), dim=1
        )

        ego_features["agent_availability"] = torch.unsqueeze(ego_features["agent_availability"], dim=1)
        agents_past_avail = torch.cat((ego_features["agent_availability"],ego_features["other_agents_availability"]), dim=1)

        static_keys = ["lanes_mid", "crosswalks"]
        disable_lane_boundaries = True
        if not disable_lane_boundaries:
            static_keys += ["lanes"]
        avail_keys = [f"{k}_availability" for k in static_keys]

        max_num_vectors = max([ego_features[key].shape[-2] for key in static_keys])

        static_polys = torch.cat([pad_points(ego_features[key], max_num_vectors) for key in static_keys], dim=1)
        mask = (torch.arange(static_polys.shape[-1]) == static_polys.shape[-1] - 1)
        static_polys[..., mask] = 0  # NOTE: this is an hack
        static_polys = static_polys
        static_avail = torch.cat([pad_avail(ego_features[key], max_num_vectors) for key in avail_keys], dim=1)

        vel_x = agents_past_polys[:, :, :, 0]
        vel_y = agents_past_polys[:, :, :, 1]
        ego_yaw = agents_past_polys[:, 0, :, 2]
        agent_dist = agents_past_polys[:, 1:, :, 2]
        mean_x = stats[0]
        mean_y = stats[1]
        std_x = stats[2]
        std_y = stats[3]
        mean_yaw = stats[4]
        std_yaw = stats[5]
        norm_vel_x = self.normalize(vel_x, mean_x, std_x)
        norm_vel_y = self.normalize(vel_y, mean_y, std_y)
        norm_yaw = self.normalize(ego_yaw, mean_yaw, std_yaw)
        norm_agent_dist = self.normalize(agent_dist, mean_yaw, 33.2631) # in normalization, mean was not really used, this is just a dummy value for the function

        norm_yaw = norm_yaw.unsqueeze(1)
        norm_yaw_dist = torch.cat((norm_yaw, norm_agent_dist), dim = 1)
        agents_past_polys = torch.cat((norm_vel_x.unsqueeze(3), norm_vel_y.unsqueeze(3), norm_yaw_dist.unsqueeze(3)), dim=3)
        agents_polys_feats = pad_points(agents_past_polys, max_num_vectors)
        agents_avail = pad_avail(agents_past_avail, max_num_vectors)

        # standardize inputs
        # agents_norm_param[0] = velocity_x
        # agents_norm_params = np.array((self.sim_data_mean_velocity, self.sim_data_std_velocity, self.sim_data_mean_yaw,self.sim_data_std_yaw ))
        # normalizing vector borrowed from urban driver code. not sure how they got those values
        # but keeping them as is
        static_norm_vector = np.array(([33.2631, 21.3976, 1.5490]))
        static_norm_vector = torch.tensor(static_norm_vector, dtype=torch.float32).to(self.device)
        static_polys_feats = static_polys / static_norm_vector

        all_polys = torch.cat([agents_polys_feats, static_polys_feats], dim=1)
        all_avail = torch.cat([agents_avail, static_avail], dim=1)
        all_avail = all_avail.bool()

        # # Embed inputs, calculate positional embedding, call local subgraph
        #all_embs, invalid_polys = self.embed_polyline(all_polys, all_avail)
        return all_polys, all_avail
        #return all_embs, invalid_polys


    def normalize(self, value, mean, std):
        normalized_value = value/ std
        # normalized_value[value<0] = - normalized_value[value<0]
        return normalized_value