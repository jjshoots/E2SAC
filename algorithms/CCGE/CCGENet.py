import torch
import torch.nn as nn
import torch.nn.functional as F
from wingman import NeuralBlocks


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, act_size, obs_size):
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size

        _features_description = [obs_size, 512, 512, act_size * 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.net(states).reshape(*states.shape[:-1], 2, self.act_size)

        if len(output.shape) > 2:
            output = output.moveaxis(-2, 0)

        return output


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, act_size, obs_size):
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size

        _features_description = [act_size + obs_size, 512, 512, 2]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        self.register_buffer("uncertainty_bias", torch.rand(1) * 100.0, persistent=True)

    def forward(self, states, actions):
        if len(actions.shape) != len(states.shape):
            states = torch.stack([states] * actions.shape[0], dim=0)

        output = torch.cat((states, actions), dim=-1)
        output = self.net(output)

        value, uncertainty = torch.split(output, 1, dim=-1)

        uncertainty = F.softplus(uncertainty + self.uncertainty_bias)

        return torch.stack((value, uncertainty), dim=0)
