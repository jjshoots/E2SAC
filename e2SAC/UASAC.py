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

    def __init__(self, num_actions, num_networks=2):
        super().__init__()

        networks = [UASACNet.Critic(num_actions) for _ in range(num_networks)]
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

    def __init__(self, num_actions):
        super().__init__()
        self.net = UASACNet.Actor(num_actions)

    def forward(self, states):
        output = torch.tanh(self.net(states))
        return output[0] * 2.0, output[1] * 10.0

    @staticmethod
    def sample(mu, sigma):
        """
        output:
            actions is of shape B x num_actions
            entropies is of shape B x 1
        """
        # lower bound sigma and bias it
        normals = dist.Normal(mu, func.softplus(sigma + 1) + 1e-6)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate entropies
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = log_probs.sum(dim=-1, keepdim=True)

        return actions, entropies

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
        entropy_tuning=True,
        target_entropy=None,
        confidence_scale=3.0,
        sup_lambda=10.0,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.use_entropy = entropy_tuning
        self.confidence_scale = confidence_scale
        self.sup_lambda = sup_lambda

        # actor head
        self.actor = GaussianActor(num_actions)

        # twin delayed Q networks
        self.critic = Q_Ensemble(num_actions)
        self.critic_target = Q_Ensemble(num_actions).eval()

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
        q = torch.std(q).detach()
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
            next_actions, entropies = self.actor.sample(*output)

            # get the next q lists and get the value, then...
            next_q_output = self.critic_target(next_states, next_actions)
            next_q = next_q_output[0]

            # ...take the min at the cat dimension
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # TD learning, targetQ = R + dones * (gamma*nextQ + entropy)
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * entropies + gamma * next_q) * dones
            )

            # update the q_std
            self.update_q_std(target_q)

        # get total difference and max over normalized total errors
        total_error = (current_q - target_q)
        u_target = total_error.abs().detach() / self.q_std
        u_target, _ = torch.max(u_target, dim=-1, keepdim=True)

        # critic loss is mean squared TD errors
        q_loss = (total_error ** 2).mean()

        # uncertainty loss is distance to predicted normalized total error
        u_loss = ((u_target - current_u) ** 2).mean()

        critic_loss = q_loss + u_loss

        log = dict()
        log["q_std"] = self.q_std
        log["u_std"] = u_loss.std().mean()
        log["u_mean"] = u_loss.mean().detach()

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

        # expectations of Q with clipped double Q
        q = self.critic(states, actions)[0]
        q, _ = torch.min(q, dim=-1, keepdim=True)

        # reinforcement target is maximization of (Q + alpha * entropy) * done
        if self.use_entropy:
            rnf_loss = -((q - self.log_alpha.exp().detach() * entropies) * dones)
        else:
            rnf_loss = -(q * dones)

        # supervisory loss is difference between predicted and label
        sup_loss = func.mse_loss(labels, actions, reduction="none")
        sup_loss *= self.sup_lambda

        # get estimate of uncertainty
        with torch.no_grad():
            uncertainty = self.critic(states, labels)[1]

            # inverse exponential
            # sup_scale = 1.0 - torch.exp(-self.confidence_scale * uncertainty)

            # tanh
            sup_scale = self.confidence_scale * uncertainty ** 2
            sup_scale = torch.tanh(sup_scale)

        rnf_loss = ((1.0 - sup_scale) * rnf_loss).mean()
        sup_loss = (sup_scale * sup_loss).mean()
        actor_loss = rnf_loss + sup_loss

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
        _, entropies = self.actor.sample(*output)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = (
            self.log_alpha * (self.target_entropy - entropies).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self.log_alpha.item()
        log["mean_entropy"] = entropies.mean().detach()

        return entropy_loss, log
