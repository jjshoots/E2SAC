#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

import ESDDQN.ESDDQNNet as ESDDQNNet


class ESDDQN(nn.Module):
    """
    Double Deep Q Network with Delayed Target Network
    """

    def __init__(
        self,
        num_actions,
        state_size,
        confidence_lambda=10.0,
        supervision_lambda=10.0,
        exploration_epsilon=0.05,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.state_size = state_size
        self.confidence_lambda = confidence_lambda
        self.supervision_lambda = supervision_lambda
        self.exploration_epsilon = exploration_epsilon
        self.num_networks = 1

        # twin delayed Q networks
        self.q = ESDDQNNet.Q_Ensemble(
            num_actions, state_size, num_networks=self.num_networks
        )
        self.q_target = ESDDQNNet.Q_Ensemble(
            num_actions, state_size, num_networks=self.num_networks
        ).eval()

        # copy weights and disable gradients for the target network
        self.q_target.load_state_dict(self.q.state_dict())
        for param in self.q_target.parameters():
            param.requires_grad = False

    def forward(self, states):
        return self.q.forward(states)

    def sample(self, q, uncertainty):
        if np.random.random_sample() < self.exploration_epsilon:
            return torch.randint(high=self.num_actions, size=q.shape[:-2]).squeeze()
        else:
            return self.infer(q, uncertainty)

    def infer(self, q, uncertainty):
        # sum over all networks
        q = q.sum(dim=-1)

        # get max over all actions
        return torch.argmax(q, dim=-1, keepdim=True)

    def update_q_target(self, tau=0.005):
        # polyak averaging update for target q network
        for target, source in zip(self.q_target.parameters(), self.q.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_loss(
        self, states, actions, rewards, next_states, dones, labels, gamma=0.99
    ):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        dones = 1.0 - dones
        actions = actions.to(dtype=torch.int64)

        # expand some stuff over the networks dim
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        # current Q and U, output is batch_size x num_actions x num_networks
        q_output, u_output = self.q(states)

        """PRECOMPUTE"""
        with torch.no_grad():
            # get the next q and u lists and get the value, then...
            next_q, next_u = self.q_target(next_states)

            # take mean over networks
            next_q = next_q.mean(dim=-1, keepdim=True)
            next_u = next_u.mean(dim=-1, keepdim=True)

            # select next actions, and expand over networks dim
            next_a = self.infer(next_q, next_u).unsqueeze(-1)

            # ... for q and u, gather from next action
            next_q = next_q.gather(dim=-2, index=next_a)
            next_u = next_u.gather(dim=-2, index=next_a)

            # Q_target = reward + dones * gamma * next_q
            target_q = rewards + gamma * next_q * dones

        # aggregate current_q and current_u for those which we only have actions for
        current_q = q_output.gather(dim=-2, index=actions)
        current_u = u_output.gather(dim=-2, index=actions)

        # compute bellman error
        bellman_error = abs(current_q - target_q)

        """Q LOSS CALCULATION"""
        # compare predictions with targets to form loss
        q_loss = (bellman_error ** 2).mean()

        """U LOSS CALCULATION"""
        # U_target = bellman_error + dones * gamma * next_u
        target_u = bellman_error.detach() + gamma * next_u * dones

        # compute uncertainty loss
        u_loss = ((current_u - target_u) ** 2).mean()

        """SUM LOSSES"""
        total_loss = q_loss + u_loss

        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["target_u"] = target_u.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        log["u_loss"] = u_loss.mean().detach()
        log["uncertainty"] = current_u.mean().detach()

        return total_loss, log
