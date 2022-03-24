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

        # twin delayed Q networks
        self.q = ESDDQNNet.Q_Ensemble(num_actions, state_size, num_networks=2)
        self.q_target = ESDDQNNet.Q_Ensemble(
            num_actions, state_size, num_networks=2
        ).eval()

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
        labels = labels.to(dtype=torch.int64)

        # current Q, output is batch_size x num_actions x num_networks
        q_output, u_output = self.q(states)

        """Q LOSS CALCULATION"""
        with torch.no_grad():
            # get the next q and u lists and get the value, then...
            next_q, next_u = self.q_target(next_states)

            # ... for q, take expectation over networks and max over actions
            next_q = next_q.mean(dim=-1, keepdim=True)[0].max(dim=-2)[0]

            # ... for u, take expectation over action and expectation over networks
            next_u = next_u.mean(dim=-2)[0].mean(dim=-1, keepdim=True)

            # Q_target = reward + dones * gamma * next_q
            target_q = rewards + gamma * next_q * dones

        # aggregate current_q and current_u for those which we only have actions for
        select = actions.unsqueeze(-2).expand(-1, -1, q_output.shape[-1])
        current_q = q_output.gather(dim=-2, index=select).squeeze(-2)
        current_u = u_output.gather(dim=-2, index=select).squeeze(-2)

        # compute bellman error
        bellman_error = abs(current_q - target_q)

        # compare predictions with targets to form loss
        q_loss = (bellman_error ** 2).mean()

        """U LOSS CALCULATION"""
        # U_target = bellman_error + dones * gamma * next_u
        bellman_error = bellman_error.mean(dim=-1, keepdim=True)
        target_u = bellman_error.detach() + gamma * next_u * dones

        # compute uncertainty loss
        u_loss = ((current_u - target_u) ** 2).mean()

        """SUPERVISION SCALE CALCULATION"""
        # get upper and lower bound
        uncertain_q = q_output + u_output
        upperbound, _ = torch.max(uncertain_q, dim=-1)
        lowerbound, _ = torch.min(uncertain_q, dim=-1)

        # compute uncertainty
        uncertainty = (
            upperbound.gather(dim=-1, index=labels)
            - lowerbound.gather(dim=-1, index=actions)
        ) / q_output.mean(-1).gather(dim=-1, index=actions).abs().detach()

        sup_scale = torch.clamp(
            uncertainty * self.confidence_lambda,
            min=0.0,
            max=1.0,
        ).detach()

        """SUPERVISION LOSS"""
        sup_loss = 0.0
        for i in range(q_output.shape[-1]):
            sup_loss = sup_loss + func.cross_entropy(
                q_output[..., i], labels.squeeze(-1), reduction="none"
            )

        sup_loss = sup_loss.unsqueeze(-1) * sup_scale * self.supervision_lambda
        sup_loss = sup_loss.mean()

        """SUM LOSSES"""
        total_loss = q_loss + u_loss + sup_loss

        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["target_u"] = target_u.mean().detach()
        log["sup_loss"] = sup_loss.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        log["u_loss"] = u_loss.mean().detach()
        log["uncertainty"] = uncertainty.mean().detach()
        log["supervision_scale"] = sup_scale.mean().detach()

        return total_loss, log
