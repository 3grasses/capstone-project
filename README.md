# Implementation of Deep Deterministic Policy Gradient (DDPG) Method on Autonomous Vehicle within the Highway-env 

*This is a collaboration with Nattapol Hirattanapan, Zelin Zheng, Vivian Ying Ying Tong, Ma Kristin Ysabel Dela Cruz, and Ziming Fang for the Data Science Capstone Project in S1 2023 at the University of Sydney, Australia.*

(add vedio)

This project is an implementation of an autonomous highway dirving algorithm based on the framework of Deep Deterministic Policy Gradient (DDPG). The simulation enivronment is created by the `highway-env` in `gymnasium` (Leurent *et al.*, 2018) under continuous action space. ***The agent is trained under an environment simulated by the highway-env package (Leurent *et al.*, 2018) in gynasium under continuous space. It is able to learn prevent collisions through iterative learning and reaches a stable state meanwhile maintainging realistic human driving behaviors.***

DDPG is a deep reinforcement learning algorithm that combines the essence of Deep Q-Network (DQN) and Deterministic Policy Gradient (DPG). It preserves the features of experience replay buffer and seperate target networks in DQN to stabilize training process and reduce data correlation. However, unlike DQN which maps the Q-value function solely, DDPG uses an Actor-Critic network to learn the Q-value function and the policy concurrently to deal with continuous action space. More detailed explanations of the theoretical background of this are provided in `DDPG-workflow.pdf` in case of your interest. In addition, ***the adoption of the deterministic policy in DDPG allows it to reuse previous experience to update the networks and calculate gradients more efficiently, which is the major difference between it and the stochastic policy of Advantage Actor Critic (A2C).*** In this project, the DDPG method is explored due to the choice of continuous action space. 

Compared to previous studies, the significance of this project includes:

- Use continuous action space to increase the complexity of the enivronment and its potential for real-world applications.
- Apply DDPG method to autonomous driving and optimize it on reward gain.
- Introduce the concept of social awareness to make the agent adhere to realistic human driving behaviors or real-world traffic regulations, such as adhereing to safe sopping distance or speed limits.

Note that the code provided here is already the version after optimization, and it is proved to outperform the A2C model in `stable-baselines` by more than 500% in terms of average test reward.

## Content

- **main.py**: The training and evaluation procedure of the algorithm.
- **ddpg.py**: The main structure of the DDPG algorithm.
- **networks.py**: The Actor and Critic networks applied in the DDPG algorithm.
- **noise.py**: The action noise used to enhance the agent's exploration
- **utils.py**: The evaluation method and other axuiliary functions, e.g., plotting. **(MOVE BUFFER PART TO DDPG)**

## Settings

#### Environment
A continuous action space which provides choices of acceleration and steering angle is adopted. However, the range of steering range is confined to [-30, 30] degree based on the real-world scenario, and the ***complete*** settings of the environment configuration can be found in `main.py`.

#### Reward Function
The reward functions are modified in response to the choice of continuous action space. Several different methods were applied to prevent ego vehicles from spinning, moving backward, or remaining in an unrealistic speed.

#### Training
The training time is 3,000 epochs with 300 maximum steps in each epoch to prevent overloading. The networks are updated at every step once the experiece replay buffer, which is of size 25000, is full. The type of exploration noise and learning rate scheduler is determined based on the optimization result, and more details can be referred to `ddpg.py`.
 
#### Evaluation
The metric used to evaluate the model perforamnce is the average reward over 30 epochs after removing regulations. The maximum step is extended to 500 epochs when evaluation for an unbiased result.

## Reference

[1] Edouard, L. (2018). *An Environment for Autonomous Driving Decision-Making.* GitHub. https://github.com/eleurent/highway-env
