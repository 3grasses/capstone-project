import gymnasium as gym
import numpy as np
import torch
import copy
import random
import statistics as stat
import warnings
warnings.filterwarnings("ignore")

import networks as network
import utils
import ddpg

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

"""
Running Instruction:
- This code module runs the DDPG model training based on the model optimisation.
- The model parameters is saved periodically during the training and can be loaded for additional training using
    ddpg_agent.load("model name"). 
- After the training is completed, it will evaluate the trained model 5 times and report the mean and standard deviation
    of the evaluation, the last evaluation trial reward plot, and the training related plots.
"""

# Set GPU if available
if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():  # Use GPU on Mac if available
    _device = "mps"
else:
    _device = "cpu"
device = torch.device(_device)
print("Run on:", device)



# Initiate the environment
env = gym.make('highway-v0', render_mode='rgb_array')

# Environment configuration
env.configure(
    {"observation": {
        "type": "Kinematics",
        "vehicles_count": 7,
        "features": ["presence", "x", "y", "vx", "vy"],
    },

        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "absolute": False,
        "lanes_count": 4,
        "reward_speed_range": [40, 60],
        "simulation_frequency": 15,
        "vehicles_count": 50,
        "policy_frequency": 10,
        "initial_spacing": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": 20,
        "collision_reward": -2,
        "action_reward": -0.3,
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.3, 0.5],
        "scaling": 7,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    })


# Start the training process
state = env.reset()

# Get state dimensions
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
state_dim_a = [env.observation_space.shape[0], env.observation_space.shape[1]]

# Get action dimension
action_dim = env.action_space.shape[0]

# Initialize Actor and Critic network
actor = network.Actor_network(state_dim_a, action_dim).to(device)
critic = network.Critic_network(state_dim_a, action_dim, 256).to(device)

# Initialize target networks
target_actor = copy.deepcopy(actor).to(device)
target_critic = copy.deepcopy(critic).to(device)

# Initialize agent
ddpg_agent = ddpg.DDPG_agent(env, actor, critic, target_actor, target_critic, device)

# Load the model parameters (To continue training on the previous trained model only)
#ddpg_agent.load("model", "model_final.pt")

# Train the agent
ddpg_agent.train()


##########################  Evaluation section #########################
# Model Evaluation
eval_reward_list = []
avg_training = []

# Evaluate the model 5 trials
for i in range (5):
    eval_reward = utils.eval_agent(ddpg_agent, env, fname="model_final.pt", device=device, load=True)
    print(f"Avg test reward over episodes for trial {i+1}: {eval_reward[1][-1]:.3f}")
    eval_reward_list.append(eval_reward)
    avg_training.append(eval_reward[1][-1])

# Show the training result plots
utils.plots(ddpg_agent)

# Show the test reward plot for the last trial
utils.plot_eval(eval_reward)

# Show the mean and standard deviation of training reward
mean_avg_reward = stat.mean(avg_training)
stdev_avg_reward = stat.stdev(avg_training)

print(f"Mean of Avg test reward: {mean_avg_reward:.3f}")
print(f"Stdev of Avg test reward: {stdev_avg_reward:.3f}")









