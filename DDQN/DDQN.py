#!/usr/bin/env python3
import warnings

import numpy as np
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

        # twin delayed Q networks
        self.q = DDQNNet.Q_Ensemble(num_actions, state_size)
        self.q_target = DDQNNet.Q_Ensemble(num_actions, state_size).eval()

        # copy weights and disable gradients for the target network
        self.q_target.load_state_dict(self.q.state_dict())
        for param in self.q_target.parameters():
            param.requires_grad = False

    def forward(self, states):
        return self.q.forward(states)

    def sample(self, q, uncertainty):
        if np.random.random_sample() < self.exploration_epsilon:
            return torch.randint(high=self.num_actions, size=[1]).squeeze()
        else:
            return self.infer(q, uncertainty)

    def infer(self, q, uncertainty):
        # sum over all networks
        q = q.sum(dim=-1)

        # assert that we only have one state
        assert q.shape[0] == 1, "sample is only supported for batch size of 1"

        return torch.argmax(q.squeeze(0))

    def update_q_target(self, tau=0.005):
        # polyak averaging update for target q network
        for target, source in zip(self.q_target.parameters(), self.q.parameters()):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_loss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """
        states is of shape B x input_shape
        actions is of shape B x num_actions
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        dones = 1.0 - dones
        actions = actions.to(dtype=torch.int64).unsqueeze(-2)

        # current Q, output is batch_size x num_actions x num_networks
        current_q, current_u = self.q(states)

        # target Q
        with torch.no_grad():
            # get the next q and u lists and get the value, then...
            next_q, next_u = self.q_target(next_states)

            # ...take the max over the action dimension
            next_q, _ = torch.max(next_q, dim=-2)

            # Q_target = reward + dones * gamma * next_q
            target_q = rewards + gamma * next_q * dones

        # aggregate current_q for those which we only have actions for
        current_q = current_q.gather(dim=-2, index=actions).squeeze(-2)
        current_u = current_u.gather(dim=-2, index=actions).squeeze(-2)

        # compute bellman error
        bellman_error = abs(current_q - target_q)

        # compare predictions with targets to form loss
        q_loss = (bellman_error ** 2).mean()
        u_loss = func.mse_loss(current_u, bellman_error.detach())

        loss = q_loss + u_loss

        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        log["u_loss"] = u_loss.mean().detach()

        return loss, log
