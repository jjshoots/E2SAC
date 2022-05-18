#!/usr/bin/env python3
import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

import SAC.SACNet as SACNet


class Q_Ensemble(nn.Module):
    """
    Q Network Ensembles
    """

    def __init__(self, num_actions, state_size, num_networks=2):
        super().__init__()

        networks = [SACNet.Critic(num_actions, state_size) for _ in range(num_networks)]
        self.networks = nn.ModuleList(networks)

    def forward(self, states):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
        output is a tuple of B x num_networks
        """
        output = []
        for network in self.networks:
            output.append(network(states).unsqueeze(-1))

        output = torch.cat(output, dim=-1)

        return output


class GaussianActor(nn.Module):
    """
    Gaussian Actor
    """

    def __init__(self, num_actions, state_size):
        super().__init__()
        self.net = SACNet.Actor(num_actions, state_size)

    def forward(self, states):
        return self.net(states)

    @staticmethod
    def sample(action_probs):
        """
        output:
            actions is of shape B x 1
        """
        cat = dist.Categorical(action_probs)
        return cat.sample()

    @staticmethod
    def infer(action_probs):
        return torch.argmax(action_probs, dim=-1, keepdim=True)


class SAC(nn.Module):
    """
    Soft Actor Critic
    """

    def __init__(
        self,
        num_actions,
        state_size,
        entropy_tuning=True,
        target_entropy=None,
        discount_factor=0.98,
        num_networks=2,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.state_size = state_size
        self.use_entropy = entropy_tuning
        self.gamma = discount_factor
        self.num_networks = 2

        # actor head
        self.actor = GaussianActor(num_actions, state_size)

        # twin delayed Q networks
        self.critic = Q_Ensemble(num_actions, state_size, num_networks)
        self.critic_target = Q_Ensemble(num_actions, state_size, num_networks).eval()

        # copy weights and disable gradients for the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        self.entropy_tuning = entropy_tuning
        if entropy_tuning:
            if target_entropy is None:
                import math
                self.target_entropy = -math.log(1 / float(num_actions)) / 4.0
            else:
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

    def calc_critic_alpha_loss(self, states, actions, rewards, next_states, dones):
        """
        states is of shape B x input_shape
        actions is of shape B x 1
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        dones = 1.0 - dones
        actions = actions.to(dtype=torch.int64)

        # current Q, output is B x num_actions x num_networks
        current_q = self.critic(states)

        # gather the relevant Q values only
        actions = actions.unsqueeze(-1).repeat(1, 1, self.num_networks)
        current_q = current_q.gather(dim=-2, index=actions).squeeze(-2)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            action_probs = self.actor(next_states)
            # actions = action_probs.argmax(dim=-1, keepdim=True)
            # actions = actions.unsqueeze(-1).repeat(1, 1, self.num_networks)

            # entropy bonus
            entropy = -(action_probs * torch.log(action_probs)).mean(dim=-1, keepdim=True)

            # get the next q lists
            next_q = self.critic_target(next_states)

            # expected next_q
            next_q = (action_probs.unsqueeze(-1) * next_q).mean(dim=-2)
            # next_q = next_q.gather(dim=-2, index=actions).squeeze(-2)
            next_q, _ = next_q.min(dim=-1, keepdim=True)

            # TD learning, targetQ = R + dones * (gamma*nextQ + entropy)
            target_q = (
                rewards
                + (self.log_alpha.exp().detach() * entropy + self.gamma * next_q)
                * dones
            )

        # critic loss is mean squared TD errors
        critic_loss = ((current_q - target_q) ** 2).mean()

        # compute loss for alpha as well
        if self.use_entropy:
            entropy_loss = (
                self.log_alpha * (entropy - self.target_entropy).detach()
            ).mean()
            loss = critic_loss + entropy_loss
        else:
            loss = critic_loss

        # logging
        log = dict()
        log["log_alpha"] = self.log_alpha.item()
        log["mean_entropy"] = entropy.mean().detach()

        return loss, log

    def calc_actor_loss(self, states, dones):
        """
        states is of shape B x input_shape
        dones is of shape B x 1
        """
        dones = 1.0 - dones

        # We re-sample actions to calculate expectations of Q.
        action_probs = self.actor(states)

        # expectations of Q with clipped double Q
        q = self.critic(states).detach()

        # maximization objective, expectations over actions, min over networks
        objective = action_probs.unsqueeze(-1) * q
        objective = objective.mean(-2)
        objective, _ = objective.min(-1, keepdim=True)

        # reinforcement target is maximization of (Q + alpha * entropy) * done
        if self.use_entropy:
            entropy = -(action_probs * torch.log(action_probs)).mean(dim=-1, keepdim=True)
            rnf_loss = -((objective + self.log_alpha.exp().detach() * entropy) * dones)
        else:
            rnf_loss = -(objective * dones)

        actor_loss = rnf_loss.mean()

        log = dict()

        return actor_loss, log
