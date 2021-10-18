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

        _features_description = [28, 128, 128]
        _activation_description = ['lrelu'] * (len(_features_description) - 1)
        self.net = Neural_blocks.generate_linear_stack(_features_description, _activation_description)


    def forward(self, state):
        return self.net(state)


class ActorHead(nn.Module):
    """
    Actor Head for Net
    """
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        _features_description = [128, 128, num_actions * 4]
        _activation_description = ['lrelu'] * (len(_features_description) - 2) + ['identity']
        self.net = Neural_blocks.generate_linear_stack(_features_description, _activation_description)


    def forward(self, states):
        states = self.net(states).reshape(-1, 4, self.num_actions).permute(1, 0, 2)

        mu, lognu, logalpha, logbeta = torch.split(states, 1, dim=0)

        nu = F.softplus(lognu) + 1e-6
        alpha = F.softplus(logalpha) + 1. + 1e-6
        beta = F.softplus(logbeta) + 1e-6

        return torch.cat([mu, nu, alpha, beta], dim=0)


class CriticHead(nn.Module):
    """
    Critic Head for Net
    """
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

        _features_description = [128, 64]
        _activation_description = ['lrelu'] * (len(_features_description) - 1)
        self.context = Neural_blocks.generate_linear_stack(_features_description, _activation_description)

        _features_description = [num_actions, 64]
        _activation_description = ['lrelu'] * (len(_features_description) - 1)
        self.action = Neural_blocks.generate_linear_stack(_features_description, _activation_description)

        _features_description = [64, 64, 1]
        _activation_description = ['lrelu'] * (len(_features_description) - 2) + ['identity']
        self.merge = Neural_blocks.generate_linear_stack(_features_description, _activation_description, batch_norm=False)


    def forward(self, states, actions):
        return self.merge(self.context(states) + self.action(actions))
