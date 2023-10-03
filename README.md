# Implementation of Deep Deterministic Policy Gradient (DDPG) Method on Autonomous Vehicle within the Highway-env 

*This is a collaboration with Nattapol Hirattanapan, Zelin Zheng, Vivian Ying Ying Tong, Ma Kristin Ysabel Dela Cruz, and Ziming Fang for the Data Science Capstone Project in S1 2023 at the University of Sydney, Australia.*

<br>

<p align="center">
<img src="https://github.com/3grasses/capstone-project/assets/146526540/56ad4c01-73b2-49ee-acc0-6e5ee5473c6f">
</p>

<br>

This project is an implementation of an autonomous highway dirving algorithm based on the framework of Deep Deterministic Policy Gradient (DDPG). The agent is trained under an environment simulated by the highway-env package (Leurent *et al.*, 2018) in gynasium and learns to avoid collisions through iterative learning meanwhile maintain realistic human driving behaviors.

DDPG is a deep reinforcement learning algorithm that combines the essence of Deep Q-Network (DQN) and Deterministic Policy Gradient (DPG). It preserves the features of experience replay buffer and seperate target networks in DQN to stabilize training process and reduce data correlation. However, unlike DQN which maps the Q-value function solely, DDPG uses an Actor-Critic network to learn the Q-value function and the policy concurrently to deal with the continuous action space. More detailed explanations of the theoretical background of this are provided in `DDPG-workflow.pdf` in case of your interest. In addition, the adoption of the deterministic policy in DDPG allows it to reuse previous experience to update the networks and accelearate the learning if a continuous action space is used, which is an advantage over the stochastic policy used in other Actor-Critic methods like Advantage Actor Critic (A2C). In this project, the DDPG method is explored due to the choice of continuous action space. 

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
A continuous action space which provides the choices of acceleration and steering angle is adopted. However, the range of steering range is confined to [-30, 30] degree based on the real-world scenario. Vluase of other elements, such as lane number or traffic density, are directly defined using the environment configuration provided by the `highway-env` and can be found in `main.py`.

#### Reward function
Due to the choice of the continuous action space, different strategies are applied to prevent the agent from spinning, moving backward, or taking other unexpected actions and encourage it to adhere to the real-world regulations and diriving behaviors as mentioned above.

#### Training
The entire training takes 3,000 epochs with 300 maximum steps in each epoch to prevent overload. The networks are updated every step once the experiece replay buffer, which has a size of 25000, is full. The type of exploration noise and learning rate scheduler are determined based on the optimization result, and the choices of the relevant parameters are stated in `ddpg.py`.
 
#### Evaluation
Mean of average test reward is used to evaluate the model performance in this project. It is calculated by taking the mean of the average reward obtained under each epoch for 30 runs. Note that the maximum step in each epoch is extended to 500 during evaluation in order to obtain a more generalized result.

## Reference

[1] Edouard, L. (2018). *An Environment for Autonomous Driving Decision-Making.* GitHub. https://github.com/eleurent/highway-env
