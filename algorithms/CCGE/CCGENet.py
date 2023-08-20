import torch
import torch.nn as nn
import torch.nn.functional as F
from wingman import NeuralBlocks


class Backbone(nn.Module):
    """Backbone Network and logic"""

    def __init__(self, embedding_size, obs_att_size, obs_img_size):
        super().__init__()

        # processes the drone attitude
        _features_description = [
            obs_att_size,
            embedding_size,
            embedding_size,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.attitude_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # process the visual input
        _channels_description = [obs_img_size[0], 32, 32, 64, 64, embedding_size]
        _kernels_description = [3] * (len(_channels_description) - 1)
        _pooling_description = [2] * (len(_channels_description) - 1)
        _activation_description = ["relu"] * (len(_channels_description) - 1)
        self.visual_net = NeuralBlocks.generate_conv_stack(
            _channels_description,
            _kernels_description,
            _pooling_description,
            _activation_description,
        )

    def forward(self, obs_att, obs_img):
        # normalize the observation image
        obs_img = (obs_img - 127.0) / 127.0

        # compute the drone attitude
        att_output = self.attitude_net(obs_att)
        img_output = self.visual_net(obs_img).view(*obs_img.shape[:-3], -1)

        return att_output, img_output


class Actor(nn.Module):
    """
    Actor network
    """

    def __init__(self, act_size, obs_att_size, obs_img_size):
        super().__init__()

        self.act_size = act_size
        embedding_size = 128

        self.backbone_net = Backbone(embedding_size, obs_att_size, obs_img_size)

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

    def forward(self, obs_att, obs_img):
        # pass things through the backbone
        att_output, img_output = self.backbone_net(obs_att, obs_img)

        # concatenate the stuff together and get the action
        output = torch.cat([att_output, img_output], dim=-1)
        output = self.merge_net(output).reshape(*obs_att.shape[:-1], 2, self.act_size)

        if len(output.shape) > 2:
            output = output.moveaxis(-2, 0)

        return output


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, act_size, obs_att_size, obs_img_size):
        super().__init__()

        embedding_size = 256

        self.backbone_net = Backbone(embedding_size, obs_att_size, obs_img_size)

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

    def forward(self, obs_att, obs_img, actions):
        # pass things through the backbone
        att_output, img_output = self.backbone_net(obs_att, obs_img)

        # if we have multiple actions, stack the observations
        if len(actions.shape) != len(obs_att.shape):
            att_output = torch.stack([att_output] * actions.shape[0], dim=0)
            img_output = torch.stack([img_output] * actions.shape[0], dim=0)

        # get the actions output
        actions = self.action_net(actions)

        # process everything
        output = torch.cat([att_output, img_output, actions], dim=-1)
        output = self.merge_net(output)

        value, uncertainty = torch.split(output, 1, dim=-1)

        uncertainty = F.softplus(uncertainty + self.uncertainty_bias)

        return torch.stack((value, uncertainty), dim=0)
