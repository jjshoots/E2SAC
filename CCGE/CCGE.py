#!/usr/bin/env python3
import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

import CCGE.CCGENet as CCGENet


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles with uncertainty estimates
    """

    def __init__(self, act_size, obs_size, num_networks=2):
        super().__init__()

        networks = [CCGENet.Critic(act_size, obs_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states, actions):
        """
        states is of shape B x input_shape
        actions is of shape B x act_size
        output is a tuple of 2 x B x num_networks
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
        self.net = CCGENet.Actor(act_size, obs_size)

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


class CCGE(nn.Module):
    """
    Critic Confidence Guided Exploration
    """

    def __init__(
        self,
        act_size,
        obs_size,
        entropy_tuning=True,
        target_entropy=None,
        discount_factor=0.99,
        confidence_lambda=10.0,
        supervision_lambda=10.0,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.act_size = act_size
        self.use_entropy = entropy_tuning
        self.gamma = discount_factor
        self.confidence_lambda = confidence_lambda
        self.supervision_lambda = supervision_lambda

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

    def update_q_target(self, tau=0.02):
        # polyak averaging update for target q network
        for target, source in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_sup_scale(self, states, actions, labels):
        # stack actions and labels to perform inference on both together
        actions_labels = torch.stack((actions, labels), dim=0)

        # put all actions and labels and states through critic
        # shape is 2 x 2 x B x num_networks,
        # value_uncertainty x actions_labels x batch x num_networks
        critic_output = self.critic(states, actions_labels)

        # splice the output to get q of actions
        expected_q = critic_output[0, 0, ...]

        """ SUPERVISION SCALE DERIVATION """
        # uncertainty is upper bound difference between suboptimal and learned
        uncertainty = (
            (
                critic_output[0, 1, ...].mean(dim=-1, keepdim=True)
                + critic_output[1, 1, ...].max(dim=-1, keepdim=True)[0]
            )
            - (
                critic_output[0, 0, ...].mean(dim=-1, keepdim=True)
                + critic_output[1, 0, ...].min(dim=-1, keepdim=True)[0]
            )
        ).detach()

        # normalize uncertainty
        uncertainty = (
            (uncertainty / critic_output[0, 0, ...].mean(dim=-1, keepdim=True).abs())
            * self.confidence_lambda
        ).detach()

        # supervision scale is a switch
        sup_scale = (uncertainty > 0.5) * 1.0

        log = dict()
        log["uncertainty"] = uncertainty.mean().detach()

        return sup_scale, expected_q, log

    def calc_critic_loss(self, states, actions, rewards, next_states, terms):
        """
        states is of shape B x input_shape
        actions is of shape B x act_size
        rewards is of shape B x 1
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # current Q and U
        current_q, current_u = self.critic(states, actions)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_states)
            next_actions, log_probs = self.actor.sample(*output)

            # get the next q and u lists and get the value, then...
            next_q, next_u = self.critic_target(next_states, next_actions)

            # ...take the min among ensembles
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)
            next_u, _ = torch.min(next_u, dim=-1, keepdim=True)

            # Q_target = reward + dones * (gamma * next_q + entropy_bonus)
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * log_probs + self.gamma * next_q)
                * terms
            )

        # calculate bellman error and take expectation over all networks
        bellman_error = (current_q - target_q) ** 2
        bellman_error = bellman_error.mean(dim=-1, keepdim=True)

        # q loss is just bellman error
        q_loss = bellman_error.mean()

        # U_target = sqrt(bellman_error + next_u^2)
        target_u = (bellman_error.detach() + (self.gamma * next_u * terms) ** 2).sqrt()
        u_loss = ((current_u - target_u) ** 2).mean()

        # critic loss is q loss plus uncertainty loss, scale losses to have the same mag
        critic_loss = q_loss + u_loss

        log = dict()
        log["bellman_error"] = bellman_error.mean().detach()
        log["target_u"] = target_u.mean().detach()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        log["u_loss"] = u_loss.mean().detach()

        return critic_loss, log

    def calc_actor_loss(self, states, terms, labels):
        """
        states is of shape B x input_shape
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(states)
        actions, entropies = self.actor.sample(*output)

        # compute supervision scale and expected q for actions
        sup_scale, expected_q, to_log = self.calc_sup_scale(states, actions, labels)

        """ REINFORCEMENT LOSS """
        # expectations of Q with clipped double Q
        expected_q, _ = torch.min(expected_q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(expected_q * terms)

        """ SUPERVISION LOSS"""
        # supervisory loss is difference between predicted and label
        sup_loss = func.mse_loss(labels, actions, reduction="none")
        sup_loss *= self.supervision_lambda

        """ ENTROPY LOSS"""
        # entropy calculation
        if self.use_entropy:
            ent_loss = self.log_alpha.exp().detach() * entropies * terms
            ent_loss = ent_loss.mean()
        else:
            ent_loss = 0.0

        """ TOTAL LOSS DERIVATION"""
        # convex combo
        rnf_loss = ((1.0 - sup_scale) * rnf_loss).mean()
        sup_loss = (sup_scale * sup_loss).mean()

        # sum the losses
        actor_loss = rnf_loss + sup_loss + ent_loss

        log = dict()
        log["sup_scale"] = sup_scale.mean().detach()
        log["sup_scale_std"] = sup_scale.std().detach()
        log = {**log, **log}

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