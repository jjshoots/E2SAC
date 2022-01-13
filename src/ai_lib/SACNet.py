import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_lib.neural_blocks import *


class Backbone(nn.Module):
    """
    Backbone for Net
    """

    def __init__(self):
        super().__init__()

        channels = [12, 128, 128, 128, 4]
        kernels = [3] * (len(channels) - 1)
        pooling = [2] * (len(channels) - 1)
        activation = ["lrelu"] * len(kernels)
        self.net = Neural_blocks.generate_conv_stack(
            channels, kernels, pooling, activation, norm="non"
        )

    def forward(self, state):
        return self.net(state).flatten(1, -1)


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        self.backbone = Backbone()

        _features_description = [64, num_actions * 2]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.backbone(states)
        output = self.net(output).reshape(-1, 2, self.num_actions).permute(1, 0, 2)

        return output


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        self.backbone = Backbone()

        _features_description = [num_actions, 64]
        _activation_description = ["identity"] * (len(_features_description) - 1)
        self.action = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

        _features_description = [64, 256, 256, 1]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.merge = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states, actions):
        states = self.backbone(states)
        return self.merge(states + self.action(actions))
