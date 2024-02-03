import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func
from wingman import NeuralBlocks


class Backbone(nn.Module):
    """Backbone Network and logic"""

    def __init__(self, embedding_size, obs_size):
        super().__init__()

        # process the visual input
        _channels_description = [obs_size[0], 16, 32, 64, embedding_size // 16]
        _kernels_description = [3] * (len(_channels_description) - 1)
        _pooling_description = [2] * (len(_channels_description) - 1)
        _activation_description = ["relu"] * (len(_channels_description) - 1)
        self.visual_net = NeuralBlocks.generate_conv_stack(
            _channels_description,
            _kernels_description,
            _pooling_description,
            _activation_description,
        )

    def forward(self, obs) -> torch.Tensor:
        # normalize the observation image
        obs = (obs - 0.5) * 2.0

        return self.visual_net(obs).view(*obs.shape[:-3], -1)


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, act_size, obs_size):
        super().__init__()

        embedding_size = 256

        self.backbone_net = Backbone(embedding_size, obs_size)

        # gets embeddings from actions
        _features_description = [act_size, embedding_size]
        _activation_description = ["relu"] * (len(_features_description) - 1)
        self.action_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

        # outputs the action after all the compute before it
        _features_description = [
            2 * embedding_size,
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

    def forward(self, obs, actions):
        # pass things through the backbone
        img_output = self.backbone_net(obs)

        # get the actions output
        act_output = self.action_net(actions)

        # process everything
        output = torch.cat([img_output, act_output], dim=-1)
        output = self.merge_net(output)

        value, uncertainty = torch.split(output, 1, dim=-1)

        uncertainty = func.softplus(uncertainty + self.uncertainty_bias)

        return torch.stack((value, uncertainty), dim=0)


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles with uncertainty estimates
    """

    def __init__(self, act_size, obs_size, num_networks=2):
        super().__init__()

        networks = [Critic(act_size, obs_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, obs, actions):
        """
        obs is of shape B x input_shape
        actions is of shape B x act_size
        output is a tuple of 2 x B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(obs, actions))

        output = torch.cat(output, dim=-1)

        return output


class GaussianActor(nn.Module):
    """
    Actor network
    """

    def __init__(self, act_size, obs_size):
        super().__init__()

        self.act_size = act_size
        embedding_size = 128

        self.backbone_net = Backbone(embedding_size, obs_size)

        # outputs the action after all the compute before it
        _features_description = [
            embedding_size,
            embedding_size,
            act_size * 2,
        ]
        _activation_description = ["relu"] * (len(_features_description) - 2) + [
            "identity"
        ]
        self.merge_net = NeuralBlocks.generate_linear_stack(
            _features_description, _activation_description
        )

    def forward(self, obs):
        """
        input:
            obs of shape B x C x H x W
        output:
            tensor of shape [2, B, *action_size] for mu and sigma
        """
        # pass things through the backbone
        output = (
            self.merge_net(self.backbone_net(obs))
            .reshape(obs.shape[0], 2, self.act_size)
            .moveaxis(-2, 0)
        )

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
