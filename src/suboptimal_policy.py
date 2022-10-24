import torch
import torch.nn as nn
from wingman import NeuralBlocks


class Suboptimal_Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, act_size, obs_size, sub_size):
        super().__init__()
        self.act_size = act_size
        self.obs_size = obs_size

        _features_description = [
            obs_size,
            sub_size,
            sub_size,
            act_size * 2,
        ]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.net(states).reshape(-1, 2, self.act_size).permute(1, 0, 2)

        return output[0], output[1]
