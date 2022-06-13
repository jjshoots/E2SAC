import torch
import torch.nn as nn
import torch.nn.functional as func

from utils.neural_blocks import Neural_blocks


class Backbone(nn.Module):
    """
    Backbone for Net
    """

    def __init__(self):
        super().__init__()

        channels = [12, 256, 256, 256, 16]
        kernels = [3] * (len(channels) - 1)
        pooling = [2] * (len(channels) - 1)
        activation = ["lrelu"] * len(kernels)
        self.net = Neural_blocks.generate_conv_stack(
            channels, kernels, pooling, activation, norm="non"
        )

    def forward(self, state):
        return self.net(state).flatten(1, -1)


class SuboptimalActor(nn.Module):
    """
    Actor network
    """

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        self.backbone = Backbone()

        _features_description = [256, 256, num_actions * 2]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.backbone(states)
        output = self.net(output).reshape(*output.shape[:-1], 2, self.num_actions)
        output = torch.movedim(output, -2, 0)

        return output