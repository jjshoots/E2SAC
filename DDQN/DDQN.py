#!/usr/bin/env python3
import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

import DDQN.DDQNNet as DDQNNet


class DDQN(nn.Module):
    """
    Double Deep Q Network with Delayed Target Network
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

        # twin delayed Q networks
        self.q = DDQNNet.Q_Ensemble(num_actions, state_size)
        self.q_target = DDQNNet.Q_Ensemble(num_actions, state_size).eval()

        # copy weights and disable gradients for the target network
        self.q_target.load_state_dict(self.q.state_dict())
        for param in self.q_target.parameters():
            param.requires_grad = False

    def update_q_target(self, tau=0.005):
        # polyak averaging update for target q network
        for target, source in zip(
            self.q_target.parameters(), self.q.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def forward(self, states):
        q_val, uncertainty = self.q.forward(states)

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
        current_q, current_u = self.q(states, actions)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_states)
            next_actions, log_probs = self.actor.sample(
                *output, n_samples=self.n_var_samples
            )

            # get the next q and u lists and get the value, then...
            next_q, next_u = self.q_target(next_states, next_actions)

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
