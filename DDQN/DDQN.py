#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn

import DDQN.DDQNNet as DDQNNet


class DDQN(nn.Module):
    """
    Double Deep Q Network with Delayed Target Network
    """

    def __init__(
        self,
        num_actions,
        state_size,
        exploration_epsilon=0.05,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.state_size = state_size
        self.exploration_epsilon = exploration_epsilon

        # twin delayed Q networks
        self.q = DDQNNet.Q_Ensemble(num_actions, state_size, num_networks=2)
        self.q_target = DDQNNet.Q_Ensemble(
            num_actions, state_size, num_networks=2
        ).eval()

        # copy weights and disable gradients for the target network
        self.q_target.load_state_dict(self.q.state_dict())
        for param in self.q_target.parameters():
            param.requires_grad = False

    def forward(self, states):
        return self.q.forward(states)

    def sample(self, q):
        if np.random.random_sample() < self.exploration_epsilon:
            return torch.randint(high=self.num_actions, size=[1]).squeeze()
        else:
            return self.infer(q)

    def infer(self, q):
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
        actions = actions.to(dtype=torch.int64)

        # current Q, output is batch_size x num_actions x num_networks
        q_output = self.q(states)

        """Q LOSS CALCULATION"""
        with torch.no_grad():
            # get the next q and u lists and get the value, then...
            next_q = self.q_target(next_states)

            # ... for q, take expectation over networks and max over actions
            next_q = next_q.mean(dim=-1, keepdim=True)[0].max(dim=-2)[0]

            # Q_target = reward + dones * gamma * next_q
            target_q = rewards + gamma * next_q * dones

        # aggregate current_q and current_u for those which we only have actions for
        select = actions.unsqueeze(-2).expand(-1, -1, q_output.shape[-1])
        current_q = q_output.gather(dim=-2, index=select).squeeze(-2)

        # compute bellman error
        bellman_error = abs(current_q - target_q)

        # compare predictions with targets to form loss
        q_loss = (bellman_error ** 2).mean()

        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = q_loss.mean().detach()

        return q_loss, log
