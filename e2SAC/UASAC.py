#!/usr/bin/env python3
import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

import e2SAC.UASACNet as UASACNet


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles with uncertainty estimates
    """

    def __init__(self, num_actions, state_size, num_networks=2):
        super().__init__()

        networks = [UASACNet.Critic(num_actions, state_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states, actions):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
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

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.net = UASACNet.Actor(num_actions, state_size)

    def forward(self, states):
        output = torch.tanh(self.net(states))
        return output[0] * 2.0, output[1] * 10.0

    @staticmethod
    def sample(mu, sigma, n_samples=1):
        """
        output:
            actions is of shape B x num_actions
            entropies is of shape B x 1
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mu, func.softplus(sigma + 1) + 1e-6)

        # sample from dist
        if n_samples > 1:
            mu_samples = normals.rsample([n_samples])
        else:
            mu_samples = normals.rsample()

        actions = torch.tanh(mu_samples)

        # calculate log_probs
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return actions, log_probs

    @staticmethod
    def infer(mu, sigma):
        return torch.tanh(mu)


class UASAC(nn.Module):
    """
    Uncertainty Aware Actor Critic
    """

    def __init__(
        self,
        num_actions,
        state_size,
        entropy_tuning=True,
        target_entropy=None,
        confidence_lambda=10.0,
        supervision_lambda=10.0,
        n_var_samples=32,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.state_size = state_size
        self.use_entropy = entropy_tuning
        self.confidence_lambda = confidence_lambda
        self.supervision_lambda = supervision_lambda
        self.n_var_samples = n_var_samples

        # actor head
        self.actor = GaussianActor(num_actions, state_size)

        # twin delayed Q networks
        self.critic = Q_Ensemble(num_actions, state_size)
        self.critic_target = Q_Ensemble(num_actions, state_size).eval()

        # copy weights and disable gradients for the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        self.entropy_tuning = entropy_tuning
        if entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -float(num_actions)
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

        # store some stuff
        self.q_std = 1e-6

    def update_q_std(self, q, tau=0.05):
        q = torch.std(q).detach() + 1e-6
        if not torch.isnan(q):
            self.q_std = (1 - tau) * self.q_std + tau * q

    def update_q_target(self, tau=0.005):
        # polyak averaging update for target q network
        for target, source in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_critic_loss(
        self, states, actions, rewards, next_states, dones, gamma=0.99
    ):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        dones = 1.0 - dones

        # current Q, output is num_networks x B x 1
        current_q, current_u = self.critic(states, actions)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_states)
            next_actions, log_probs = self.actor.sample(
                *output, n_samples=self.n_var_samples
            )

            # get the next q and u lists and get the value, then...
            next_q, next_u = self.critic_target(next_states, next_actions)

            # ...take the min at the cat dimension
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)
            next_u, _ = torch.min(next_u, dim=-1, keepdim=True)

            # Q_target = reward + dones * (gamma * next_q + entropy_bonus)
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * log_probs + gamma * next_q) * dones
            )
            target_q = target_q.mean(dim=0)

            # calculate bellman error and take expectation over all networks
            bellman_error = (current_q - target_q).abs()
            bellman_error = bellman_error.mean(dim=-1, keepdim=True)

            # calculate next_u and take expectation over all next actions
            next_u = next_u.mean(dim=0)

            # U_target = max(bellman_error) + dones * (gamma * mean(next_U))
            target_u = bellman_error + (gamma * next_u) * dones

        # compare predictions with targets to form loss
        q_loss = ((current_q - target_q) ** 2).mean()
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

    def calc_actor_loss(self, states, dones, labels):
        """
        states is of shape B x input_shape
        dones is of shape B x 1
        """
        dones = 1.0 - dones

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(states)
        actions, entropies = self.actor.sample(*output)

        # stack actions and labels to perform inference on both together
        actions_labels = torch.stack((actions, labels), dim=0)

        # put all actions and labels and states through critic
        # shape is 2 x 2 x B x num_networks,
        # value_uncertainty x actions_labels x batch x num_networks
        critic_output = self.critic(states, actions_labels)

        # splice the output to get what we want
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
            uncertainty / critic_output[0, 0, ...].mean(dim=-1, keepdim=True).abs()
        ).detach()

        # calculate supervision scale
        sup_scale = torch.clamp(
            uncertainty * self.confidence_lambda, min=0.0, max=1.0
        ).detach()

        """ REINFORCEMENT LOSS """
        # expectations of Q with clipped double Q
        expected_q, _ = torch.min(expected_q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(expected_q * dones)

        """ SUPERVISION LOSS"""
        # supervisory loss is difference between predicted and label
        sup_loss = func.mse_loss(labels, actions, reduction="none")
        sup_loss *= self.supervision_lambda

        """ ENTROPY LOSS"""
        # entropy calculation
        if self.use_entropy:
            ent_loss = self.log_alpha.exp().detach() * entropies * dones
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
        log["uncertainty"] = uncertainty.mean().detach()

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
