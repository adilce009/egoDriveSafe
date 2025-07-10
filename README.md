This repository contains the implementation of my Master's thesis project, which focuses on learning safe and efficient driving policies for autonomous vehicles 
using ** Deep Reinforcement Learning (DRL)**.

**Thesis Title: Learning Safe and Efficient Driving Policy with Proximal Policy Optimization (PPO) using Rule-Based Rewards and Real-World Offline Data** (yet to appear online)

Please note that the repository is not updated completely yet. The remaining files will be uloaded soon and a compelete Readme file will be provided as soon as 
the repository is updated fully.

The project was bulilt on the woven-planet level 5 self-driving car simulator (https://github.com/woven-planet/l5kit)
The original contribution of the thesis was the rule-based reward function that enables the self-driving car to learn a policy compatible with the basic
driving rules, such as following a safe distance to the leading vehicle and following recommended speed limit, considering the distance to the leading vehicle,
traffic light status (red/green), distance to the intersection. The reward function also takes collision and off-road incidence to avoid such undesired incidences
while driving. The necessary functions for the proposed reward function is implemented in the file **\environment\reward_vec.py**

Several l5kit base-files were modified to fit into the thesis work implementation. The details will be provided soon.

The approach involves extracting features from real-world driving data (woven planet prediction dataset). Two feature extractors were compared: a CNN based model and 
a graph-attention based model. For the two models, as obvious, two separate state representations were used: a rasterized image representation of the scene(and that was
being used with the CNN model) and a vectorized state representation of the scene (and that was being used with the graph-attention based model).
Then trained a policy network using the Proximal Policy Optimization (PPO) algorithm. Key components include:

- Feature extraction from agents and map elements (lanes, traffic lights, locations of the vehicles)
- Local subgraph construction and attention-based interaction modeling (graph-attention based model)
- Actor-Critic policy network for control (acceleration and yaw)
- Custom reward function for safe driving behavior (main contribution)

**Requirements**

- Python 3.8+
- PyTorch
- Stable-Baselines3
- NumPy, Pandas, Matplotlib, Bokeh
