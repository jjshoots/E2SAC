import torch
import torch.nn as nn
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

        embedding_size = 128

        self.backbone_net = Backbone(
            embedding_size, obs_atti_size, obs_targ_size, max_targ_length
        )

        # outputs the action after all the compute before it
        _features_description = [
            2 * embedding_size + act_size,
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
        if actions.shape[0] != obs_atti.shape[0]:
            obs_atti = torch.stack([obs_atti] * actions.shape[0], dim=0)
            obs_targ = torch.stack([obs_targ] * actions.shape[0], dim=0)

        # pass things through the backbone
        atti_output, targ_output = self.backbone_net(obs_atti, obs_targ)

        output = torch.cat([atti_output, targ_output, actions], dim=-1)
        output = self.merge_net(output)

        return output

