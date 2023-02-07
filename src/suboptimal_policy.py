import torch
import torch.nn as nn
from wingman import NeuralBlocks


class Backbone(nn.Module):
    """Backbone Network and logic"""

    def __init__(self, embedding_size, obs_atti_size, obs_targ_size, context_length):
        super().__init__()

        self.context_length = context_length

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

        # compute embeddings from target deltas
        _features_description = [obs_targ_size, int(embedding_size / 2)]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.embedding_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # processes the target deltas
        _features_description = [embedding_size, embedding_size]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.target_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # learned positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn((context_length, int(embedding_size / 2)), requires_grad=True)
        )

    def forward(self, obs_atti, obs_targ):
        # compute the drone attitude
        atti_output = self.attitude_net(obs_atti)

        # shorten the targets to only the context length
        obs_targ = obs_targ[..., : self.context_length, :]

        # expand the positional encoding if needed
        if len(obs_targ.shape) != len(self.positional_encoding.shape):
            pos_enc = torch.stack([self.positional_encoding] * obs_targ.shape[0], dim=0)
        else:
            pos_enc = self.positional_encoding

        # pass target through network, add positional encoding
        targ_output = self.embedding_net(obs_targ)
        targ_output = torch.cat((targ_output, pos_enc), dim=-1)
        targ_output = self.target_net(targ_output)

        # masking then take mean
        mask = torch.ones_like(targ_output, requires_grad=False)
        mask[obs_targ.abs().sum(dim=-1) == 0] = 0.0
        targ_output = targ_output * mask
        targ_output = targ_output.mean(dim=-2)

        return atti_output, targ_output


class Suboptimal_Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, act_size, obs_atti_size, obs_targ_size, context_length):
        super().__init__()

        self.act_size = act_size
        embedding_size = 128

        self.backbone_net = Backbone(
            embedding_size, obs_atti_size, obs_targ_size, context_length
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
