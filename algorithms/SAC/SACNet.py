import torch
import torch.nn as nn
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
        output = self.net(states).reshape(-1, 2, self.act_size).permute(1, 0, 2)

        return output


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, act_size, obs_size):
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size

        _features_description = [act_size + obs_size, 512, 512, 1]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states, actions):
        if len(actions.shape) != len(states.shape):
            states = torch.stack([states] * actions.shape[0], dim=0)

        output = torch.cat((states, actions), dim=-1)
        output = self.net(output)

        return output
