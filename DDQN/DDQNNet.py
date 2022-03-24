import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_blocks import Neural_blocks


class Q_Network(nn.Module):
    """
    Q Network with uncertainty estimates
    """

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        _features_description = [state_size, 64, 64, num_actions]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        return self.net(states)


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles with uncertainty estimates
    """

    def __init__(self, num_actions, state_size, num_networks=1):
        super().__init__()

        networks = [Q_Network(num_actions, state_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
        output is a tuple of 1 x B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(states))

        output = torch.stack(output, dim=-1)

        return output
