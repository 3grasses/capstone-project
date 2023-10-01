# Implementation of Deep Deterministic Policy Gradient (DDPG) Method on Autonomous Vehicle within the Highway-env 

*It is a collaboration with Nattapol Hirattanapan, Zelin Zheng, Vivian Ying Ying Tong, Ma Kristin Ysabel Dela Cruz, and Ziming Fang for the Data Science Capstone Project in S1 2023 at the University of Sydney, Australia.*

(add vedio)

This project is an implementation of an autonomous highway dirving algorithm based on the framework of Deep Deterministic Policy Gradient (DDPG) method. The enivronment is simulated by the `highway-env` in `gymnasium` (Leurent *et al.*, 2018) under continuous action space. **The ego vehicle is able to learn to avoid collisions through iterative learning and reaches a stable state after training.**

DDPG is a deep reinforcement learning algorithm that combines the essence of Deep Q-Network (DQN) and Deterministic Policy Gradient (DPG) but with more extensions. It preserves the features of experience replay buffer and seperate target networks in DQN to stabilize training process and reduce data correlation. However, unlike DQN which maps the Q-value function solely, DDPG uses an Actor-Critic network to learn the Q-value function and the policy concurrently to deal with continuous action space. More detailed explanations of the theoretical background of this are providied in `DDPG-workflow.md` in case of your interest. In addition, the adaption of the deterministic policy in DDPG makes it able to reuse previous experience to update the network compared to the stochastic policy of Advantage Actor Critic (A2C).

In this project, the continuous action space is adapted to make the environment closer to the reality, so the DDPG algorithm hence becomes one of the best candidates of our choice. 

The implication of this project includes:

- Use the continuous action space to make the enivronment closer to the reality.
- Adapt DDPG framework which can make the learning efficiency more efficient and imrpoved comapred to baseline model.
- Introduced the concept of social awareness to make the driver's behavior adhere to real scenarios, such as slowing down when geeting too close to front vehicles.

Note that the code provided here is already the optimized version, and it was proved to outperform the A2C model in `stable-baselines` by more than 400% in terms of the average test reward under 30 epochs.

## Content

There are five files inlcuded:

- **ddpg.py**: The main structure of the DDPG algorithm.
- **noise.py**: The exploration noise added to the action output by the algorithm.
- **networks.py**: The neural network forming the Actor and Critic networks used in the DDPG algorithm.
- **main.py**: The methods used to train and evaluate the performance of the DDPG algorithm.
- **utils.py**: Other axuliary functions, e.g., plotting, etc.

## Settings

#### Environment
The continuous action sapce adopted in this project. However, the range of steering range is confined to [-30, 30] degree based on the real-world scenario, and more details of the environment configuration can be checked in code files.

#### Training
The training epoch is 3,000 epochs with maximum steps of 300 for each epoch. This maximum limit is set to prevent the training time from becoming too long.
OU-noise was used in this algorithm based on our optimization result. Learning rate... All the parameters were determined based on our optimization results will be not revealed in this porject.
The reward functions are modified in response to the choice of continuous action space. Several different methods were applied to prevent ego vehicles from spinning, moving backward, or remaining in an unrealistic speed.

#### Evaluation
The evaluation was 30 epochs.

## Reference
