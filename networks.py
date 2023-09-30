import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic_network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        """
        Construct the critic network
        :param state_dim: The state dimension
        :param action_dim: The action dimension
        :param hidden: The hidden layer dimension
        """
        super(Critic_network, self).__init__()
        # CNN 2d
        kernel = 2
        padding = 0
        stride = 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel, stride=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=kernel, stride=1)


        out_height, out_width = calculate_conv_output(state_dim, 3, padding, stride=1)
        out_height, out_width = calculate_conv_output([out_height, out_width], kernel, padding, stride=1)
        out_height, out_width = calculate_conv_output([out_height, out_width], kernel, padding, stride=1)

        flat_dim = out_height * out_width * 128

        self.fc1 = nn.Linear(flat_dim + action_dim, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden * 2)
        self.fc3 = nn.Linear(hidden * 2, 1)


    def forward(self, state, action):
        # CNN 2d
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = torch.cat([x, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value


def calculate_conv_output(in_shape, kernel, padding=0, stride=1):
    """
    Calculate the CNN layer output dimension
    :param in_shape: The input dimension
    :param kernel: The kernel size
    :param padding: Padding on CNN
    :param stride: The stride value
    :return: A list of height and width of the output
    """

    out_height = int(((in_shape[0] + 2 * padding - kernel) / stride) + 1)
    out_width = int(((in_shape[1] + 2 * padding - kernel) / stride) + 1)
    return out_height, out_width


class Actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        """
        Construct the actor network
        :param state_dim: The state dimension
        :param action_dim: The action dimension
        :param init_w: The initial weight of the output layer
        """
        super(Actor_network, self).__init__()
        kernel = 2
        padding = 0
        stride = 1

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel, stride=stride)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=kernel, stride=stride)

        # calculate cnn output
        out_height, out_width = calculate_conv_output(state_dim, 3, padding, stride=1)
        out_height, out_width = calculate_conv_output([out_height, out_width], kernel, padding, stride=1)
        out_height, out_width = calculate_conv_output([out_height, out_width], kernel, padding, stride=1)

        self.fc1 = nn.Linear(64 * out_height * out_width, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

        # initialize the the neural network parameter to the last layer
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action = torch.tanh(self.fc3(x))
        return action
