import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
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
        _channels_description = [obs_img_size[0], 16, 32, 64, embedding_size // 16]
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
        obs_img = (obs_img - 0.5) * 2.0

        # compute the drone attitude
        att_output = self.attitude_net(obs_att)
        img_output = self.visual_net(obs_img).view(*obs_img.shape[:-3], -1)

        return att_output, img_output


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

        uncertainty = func.softplus(uncertainty + self.uncertainty_bias)

        return torch.stack((value, uncertainty), dim=0)


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles with uncertainty estimates
    """

    def __init__(self, act_size, obs_att_size, obs_img_size, num_networks=2):
        super().__init__()

        networks = [
            Critic(act_size, obs_att_size, obs_img_size) for _ in range(num_networks)
        ]
        self.networks = nn.ModuleList(networks)

    def forward(self, obs_atti, obs_targ, actions):
        """
        obs_atti, obs_targ is of shape B x input_shape
        actions is of shape B x act_size
        output is a tuple of 2 x B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(obs_atti, obs_targ, actions))

        output = torch.cat(output, dim=-1)

        return output


class GaussianActor(nn.Module):
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
        """
        input:
            obs_att of shape B x att
            obs_img of shape B x C x H x W
        output:
            tensor of shape [2, B, *action_size] for mu and sigma
        """
        # pass things through the backbone
        att_output, img_output = self.backbone_net(obs_att, obs_img)

        # concatenate the stuff together and get the action
        output = torch.cat([att_output, img_output], dim=-1)
        output = self.merge_net(output).reshape(*obs_att.shape[:-1], 2, self.act_size)

        if len(output.shape) > 2:
            output = output.moveaxis(-2, 0)

        return output

    @staticmethod
    def sample(mu, sigma):
        """
        output:
            actions is of shape B x act_size
            entropies is of shape B x 1
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mu, func.softplus(sigma) + 1e-6)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate log_probs
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return actions, log_probs

    @staticmethod
    def infer(mu, sigma):
        return torch.tanh(mu)
