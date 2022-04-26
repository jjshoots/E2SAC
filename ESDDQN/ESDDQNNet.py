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

        channels = [12, 32, 64, 128, 16]
        kernels = [3] * (len(channels) - 1)
        pooling = [2] * (len(channels) - 1)
        activation = ["lrelu"] * len(kernels)
        self.net = Neural_blocks.generate_conv_stack(
            channels, kernels, pooling, activation, norm="non"
        )

    def forward(self, state):
        return self.net(state).flatten(1, -1)


class Q_Network(nn.Module):
    """
    Q Network with uncertainty estimates
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
        output = self.net(output)

        value, uncertainty = torch.split(output, self.num_actions, dim=-1)

        uncertainty = func.softplus(uncertainty)

        return torch.stack((value, uncertainty), dim=0)


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles with uncertainty estimates
    """

    def __init__(self, num_actions, num_networks=1):
        super().__init__()

        networks = [Q_Network(num_actions) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
        output is a tuple of 2 x B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(states))

        output = torch.stack(output, dim=-1)

        return output
