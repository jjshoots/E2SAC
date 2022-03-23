import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_blocks import Neural_blocks


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        _features_description = [num_actions + state_size, 256, 256, num_actions * 2]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
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

        value, uncertainty = torch.split(output, self.num_actions, dim=-1)

        uncertainty = torch.exp(-uncertainty)

        return torch.stack((value, uncertainty), dim=0)
