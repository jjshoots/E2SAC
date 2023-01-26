#!/usr/bin/env python3
import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

from .SACNet import Actor, Critic


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles
    """

    def __init__(self, act_size, obs_size, num_networks=2):
        super().__init__()

        networks = [Critic(act_size, obs_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states, actions):
        """
        states is of shape B x input_shape
        actions is of shape B x act_size
        output is a tuple of B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(states, actions))

        output = torch.cat(output, dim=-1)

        return output


class GaussianActor(nn.Module):
    """
    Gaussian Actor
    """

    def __init__(self, act_size, obs_size):
        super().__init__()
        self.net = Actor(act_size, obs_size)

    def forward(self, states):
        output = self.net(states)
        return output[0], output[1]

    @staticmethod
    def sample(mu, sigma):
        """
        output:
            actions is of shape B x act_size
            entropies is of shape B x 1
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mu, func.softplus(sigma + 1) + 1e-6)

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


class SAC(nn.Module):
    """
    Soft Actor Critic
    """

    def __init__(
        self,
        act_size,
        obs_size,
        entropy_tuning=True,
        target_entropy=None,
        discount_factor=0.98,
    ):
        super().__init__()

        self.act_size = act_size
        self.obs_size = obs_size
        self.use_entropy = entropy_tuning
        self.gamma = discount_factor

        # actor head
        self.actor = GaussianActor(act_size, obs_size)

        # twin delayed Q networks
        self.critic = Q_Ensemble(act_size, obs_size)
        self.critic_target = Q_Ensemble(act_size, obs_size).eval()

        # copy weights and disable gradients for the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        self.entropy_tuning = entropy_tuning
        if entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -float(act_size)
            else:
                if target_entropy > 0.0:
                    warnings.warn(
                        f"Target entropy is recommended to be negative,\
                                  currently it is {target_entropy},\
                                  I hope you know what you're doing..."
                    )
                self.target_entropy = target_entropy
            self.log_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        else:
            self.log_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def update_q_target(self, tau=0.05):
        # polyak averaging update for target q network
        for target, source in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_critic_loss(self, states, actions, rewards, next_states, terms):
        """
        states is of shape B x input_shape
        actions is of shape B x act_size
        rewards is of shape B x 1
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # current Q, output is num_networks x B x 1
        current_q = self.critic(states, actions)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_states)
            next_actions, log_probs = self.actor.sample(*output)

            # get the next q lists then...
            next_q = self.critic_target(next_states, next_actions)

            # ...take the min at the cat dimension
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # TD learning, targetQ = R + dones * (gamma*nextQ + entropy)
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * log_probs + self.gamma * next_q)
                * terms
            )

        # critic loss is mean squared TD errors
        q_loss = ((current_q - target_q) ** 2).mean()

        # critic loss is q error
        critic_loss = q_loss

        log = dict()

        return critic_loss, log

    def calc_actor_loss(self, states, terms):
        """
        states is of shape B x input_shape
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(states)
        actions, entropies = self.actor.sample(*output)

        # expectations of Q with clipped double Q
        q = self.critic(states, actions)
        q, _ = torch.min(q, dim=-1, keepdim=True)

        # reinforcement target is maximization of (Q + alpha * entropy) * done
        if self.use_entropy:
            rnf_loss = -((q - self.log_alpha.exp().detach() * entropies) * terms)
        else:
            rnf_loss = -(q * terms)

        actor_loss = rnf_loss.mean()

        log = dict()

        return actor_loss, log

    def calc_alpha_loss(self, states):
        """
        states is of shape B x input_shape
        """
        if not self.entropy_tuning:
            return torch.zeros(1)

        output = self.actor(states)
        _, log_probs = self.actor.sample(*output)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -(
            self.log_alpha * (self.target_entropy + log_probs).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self.log_alpha.item()
        log["mean_entropy"] = -log_probs.mean().detach()

        return entropy_loss, log
