import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_blocks import Neural_blocks


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        _features_description = [state_size, 100, 100, num_actions * 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.net(states).reshape(-1, 2, self.num_actions).permute(1, 0, 2)

        return output


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        _features_description = [num_actions + state_size, 100, 100, 1]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states, actions):
        if len(actions.shape) != len(states.shape):
            states = torch.stack([states] * actions.shape[0], dim=0)

        output = torch.cat((states, actions), dim=-1)
        output = self.net(output)

        return output
