import torch
import torch.nn as nn

from utils.neural_blocks import Neural_blocks


class Suboptimal_Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        _features_description = [
            state_size,
            100,
            100,
            num_actions * 2,
        ]
        _activation_description = ["lrelu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.net = Neural_blocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, states):
        output = self.net(states).reshape(-1, 2, self.num_actions).permute(1, 0, 2)

        return output[0], output[1]
