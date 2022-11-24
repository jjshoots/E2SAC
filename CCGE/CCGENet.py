import torch
import torch.nn as nn
import torch.nn.functional as F
from wingman import NeuralBlocks


class Backbone(nn.Module):
    """Backbone Network and logic"""

    def __init__(self, embedding_size, obs_atti_size, obs_targ_size, max_targ_length):
        super().__init__()

        # processes the drone attitude
        _features_description = [
            obs_atti_size,
            embedding_size,
            embedding_size,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.attitude_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # processes the target deltas
        _features_description = [obs_targ_size, embedding_size]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.target_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # learned positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn((max_targ_length, embedding_size), requires_grad=True)
        )

    def forward(self, obs_atti, obs_targ):
        # compute the drone attitude
        atti_output = self.attitude_net(obs_atti)

        # pass target through network, add positional encoding, then mask, then mean
        targ_output = self.target_net(obs_targ) + self.positional_encoding
        targ_output[obs_targ.abs().sum(dim=-1) != 0] *= 0.0
        targ_output = targ_output.mean(dim=-2)

        return atti_output, targ_output


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, act_size, obs_atti_size, obs_targ_size, max_targ_length):
        super().__init__()

        self.act_size = act_size
        embedding_size = 128

        self.backbone_net = Backbone(
            embedding_size, obs_atti_size, obs_targ_size, max_targ_length
        )

        # outputs the action after all the compute before it
        _features_description = [
            2 * embedding_size,
            embedding_size,
            act_size * 2,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.merge_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, obs_atti, obs_targ):
        # pass things through the backbone
        atti_output, targ_output = self.backbone_net(obs_atti, obs_targ)

        # concatenate the stuff together and get the action
        output = torch.cat([atti_output, targ_output], dim=-1)
        output = self.merge_net(output).reshape(*obs_atti.shape[:-1], 2, self.act_size)

        if len(output.shape) > 2:
            output = output.moveaxis(-2, 0)

        return output


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, act_size, obs_atti_size, obs_targ_size, max_targ_length):
        super().__init__()

        embedding_size = 256

        self.backbone_net = Backbone(
            embedding_size, obs_atti_size, obs_targ_size, max_targ_length
        )

        # gets embeddings from actions
        _features_description = [act_size, embedding_size]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.action_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # outputs the action after all the compute before it
        _features_description = [
            3 * embedding_size,
            embedding_size,
            2,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.merge_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        self.register_buffer("uncertainty_bias", torch.rand(1) * 100.0, persistent=True)

    def forward(self, obs_atti, obs_targ, actions):
        # pass things through the backbone
        atti_output, targ_output = self.backbone_net(obs_atti, obs_targ)

        # if we have multiple actions, stack the observations
        if len(actions.shape) != len(obs_atti.shape):
            atti_output = torch.stack([atti_output] * actions.shape[0], dim=0)
            targ_output = torch.stack([targ_output] * actions.shape[0], dim=0)

        # get the actions output
        actions = self.action_net(actions)

        # process everything
        output = torch.cat([atti_output, targ_output, actions], dim=-1)
        output = self.merge_net(output)

        value, uncertainty = torch.split(output, 1, dim=-1)

        uncertainty = F.softplus(uncertainty + self.uncertainty_bias)

        return torch.stack((value, uncertainty), dim=0)
