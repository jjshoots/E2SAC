import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_blocks import Neural_blocks



class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        _features_description = [28, 32, 32, num_actions * 2]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
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

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        _features_description = [num_actions, 64]
        _activation_description = ["identity"] * (len(_features_description) - 1)
        self.action = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

        _features_description = [28, 64]
        _activation_description = ["identity"] * (len(_features_description) - 1)
        self.state = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

        _features_description = [64, 256, 2]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.merge = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states, actions):
        actions = self.action(actions)
        states = self.state(states)

        if len(actions.shape) != len(states.shape):
            states = states.unsqueeze(0)

        output = self.merge(states + actions)

        value, uncertainty = torch.split(output, 1, dim=-1)

        uncertainty = torch.exp(-uncertainty)

        return torch.stack((value, uncertainty), dim=0)
