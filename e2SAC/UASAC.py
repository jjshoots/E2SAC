#!/usr/bin/env python3
import warnings

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as func

import e2SAC.derived_normals as DN
import e2SAC.UASACNet as UASACNet


class TwinnedQNetwork(nn.Module):
    """
    Twin Q Network
    """

    def __init__(self, num_actions):
        super().__init__()

        # critic, clipped double Q
        self.Q_network1 = UASACNet.Critic(num_actions)
        self.Q_network2 = UASACNet.Critic(num_actions)

    def forward(self, states, actions):
        """
        states is of shape ** x num_inputs
        actions is of shape ** x num_actions
        output is a tuple of [** x 1], [** x 1]
        """
        # get q1 and q2
        q1 = self.Q_network1(states, actions)
        q2 = self.Q_network2(states, actions)

        return q1, q2


class GaussianActor(nn.Module):
    """
    Gaussian Actor Wrapper for Deep Evidential Regression
    """

    def __init__(self, num_actions):
        super().__init__()
        self.net = UASACNet.Actor(num_actions)

    def forward(self, states):
        return self.net(states)

    @staticmethod
    def sample(gamma, nu, alpha, beta):
        """
        output:
            actions is of shape ** x num_actions
            entropies is of shape ** x 1
            log_probs is of shape ** x num_actions
        """
        output = gamma, nu, alpha, beta
        normals = DN.Normal(*output)

        # sample from dist
        mu_samples = normals.rsample()
        actions = torch.tanh(mu_samples)

        # calculate entropies
        log_probs = normals.log_prob(mu_samples) - torch.log(1 - actions.pow(2) + 1e-6)
        entropies = log_probs.sum(dim=-1, keepdim=True)

        return actions, entropies

    @staticmethod
    def infer(gamma, nu, alpha, beta):
        return torch.tanh(gamma)


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
    ):
        super().__init__()

        self.num_actions = num_actions
        self.use_entropy = entropy_tuning
        self.confidence_scale = confidence_scale

        # actor head
        self.actor = GaussianActor(num_actions)

        # twin delayed Q networks
        self.critic = TwinnedQNetwork(num_actions)
        self.critic_target = TwinnedQNetwork(num_actions).eval()

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
        self.q_std = None
        self.sup_scale_mean = 1.0

    def update_q_std(self, q, tau=0.05):
        q = torch.std(q).detach()
        if not torch.isnan(q):
            if self.q_std is None:
                self.q_std = q
            else:
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
        states is of shape B x img_size
        actions is of shape B x 3
        rewards is of shape B x 1
        dones is of shape B x 1
        """
        dones = 1.0 - dones

        # current Q
        curr_q1, curr_q2 = self.critic(states, actions)

        # target Q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_states)
            next_actions, entropies = self.actor.sample(*output)

            next_q1, next_q2 = self.critic_target(next_states, next_actions)

            # concatenate both qs together then...
            next_q = torch.cat((next_q1, next_q2), dim=-1)

            # ...take the min at the cat dimension
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # TD learning, targetQ = R + dones * (gamma*nextQ + entropy)
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * entropies + gamma * next_q) * dones
            )

        # critic loss is mean squared TD errors
        q1_loss = func.mse_loss(curr_q1, target_q, reduction="none")
        q2_loss = func.mse_loss(curr_q2, target_q, reduction="none")
        q_loss = (q1_loss + q2_loss) / 2.0

        # update the q_std
        self.update_q_std(target_q)

        # DN regularizer scale
        regularizer = abs(curr_q1 - target_q) + abs(curr_q2 - target_q)
        regularizer = regularizer / (self.q_std + 1e-6) + 1e-6

        return q_loss.mean(), regularizer.detach()

    def calc_actor_loss(self, states, dones, labels):
        """
        states is of shape B x img_size
        dones is of shape B x 1
        labels is of shape B x 3
        """
        dones = 1.0 - dones

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(states)
        actions, entropies = self.actor.sample(*output)

        # expectations of Q with clipped double Q
        q1, q2 = self.critic(states, actions)
        q, _ = torch.min(torch.cat((q1, q2), dim=-1), dim=-1, keepdim=True)

        # reinforcement target is maximization of (Q + alpha * entropy) * done
        if self.use_entropy:
            rnf_loss = -((q - self.log_alpha.exp().detach() * entropies) * dones)
        else:
            rnf_loss = -(q * dones)

        # supervisory loss is difference between predicted and label
        sup_loss = func.mse_loss(labels, torch.tanh(output[0]), reduction='none')

        # importance sampling under derived Normal
        NLL_subopt = -DN.Normal_NLL(torch.atanh(labels), *output, reduce=False)
        NLL_policy = -DN.Normal_NLL(output[0], *output, reduce=False)
        imp_scale = (NLL_subopt - NLL_policy).exp().detach()

        # supervision scale calculation
        sup_scale = imp_scale * torch.exp(-self.confidence_scale * output[1])
        sup_scale = sup_scale.detach()

        # Derived Normal regularizer
        alpha = output[2]
        nu = output[1]

        return rnf_loss, sup_loss, sup_scale, alpha, nu

    def calc_alpha_loss(self, states):
        if not self.entropy_tuning:
            return torch.zeros(1)

        output = self.actor(states)
        _, entropies = self.actor.sample(*output)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = (
            self.log_alpha * (self.target_entropy - entropies).detach()
        ).mean()

        return entropy_loss
