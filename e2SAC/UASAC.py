#!/usr/bin/env python3
import math
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
        entropy_tuning=True,
        target_entropy=None,
        confidence_lambda=10.0,
        confidence_offset=4.0,
        supervision_lambda=10.0,
        uncertainty_skew=0.1,
        n_var_samples=32,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.use_entropy = entropy_tuning
        self.confidence_lambda = confidence_lambda
        self.confidence_offset = confidence_offset
        self.supervision_lambda = supervision_lambda
        self.uncertainty_skew = uncertainty_skew
        self.n_var_samples = n_var_samples

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
        self.q_var = 1e-6

    def update_q_var(self, q, tau=0.05):
        q = torch.var(q).detach() + 1e-6
        if not torch.isnan(q):
            self.q_var = (1 - tau) * self.q_var + tau * q

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
            # next_actions is of shape n_samples * B * num_actions
            # log_probs is of shape n_samples * B * 1
            output = self.actor(next_states)
            next_actions, log_probs = self.actor.sample(
                *output, n_samples=self.n_var_samples
            )

            # get the next q lists and get the value, then...
            # next_q is of shape n_samples * B * 2
            next_q = self.critic_target(next_states, next_actions)[0]

            # ...take the min at the cat dimension
            # next_q is of shape n_samples * B * 1
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # TD learning, targetQ = R + dones * (gamma*nextQ + entropy)
            # target_q is of shape n_samples * B * 1
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * log_probs + gamma * next_q) * dones
            )

            # update the q_var
            self.update_q_var(target_q)

        # # calculate expected loss in prediction of q
        # # take maximum amongst all networks
        # total_error = (current_q.unsqueeze(0) - target_q) ** 2
        # total_error = total_error.mean(dim=0)
        # total_error = total_error.max(dim=-1, keepdim=True)[0]

        # # aleatoric uncertainty is just variance in the target
        # # take minimum amongst all networks
        # aleatoric = target_q.var(dim=0)
        # aleatoric = aleatoric.min(dim=-1, keepdim=True)[0]

        # # epistemic uncertainty is upper bound difference between total error and aleatoric
        # epistemic = torch.clamp(total_error - aleatoric, min=0.0)

        # # u_loss is upper bound on epistemic uncertainty, skewed assymetrically, normalized
        # u_loss = (epistemic.detach() / self.q_var) - current_epistemic
        # # u_loss = func.leaky_relu(u_loss, negative_slope=self.uncertainty_skew)
        # u_loss = u_loss ** 2
        # u_loss = u_loss.mean()

        # # q_loss is just mse of total error
        # q_loss = total_error.mean()

        # calculate mean bellman error, take maximum amongst all networks
        q_loss = (current_q - target_q.mean(dim=0)) ** 2
        q_loss = q_loss.max(dim=-1, keepdim=True)[0]

        # calculate epistemic prediction error, assymetrically skew
        u_loss = (q_loss.detach() - current_u) ** 2
        # u_loss = func.leaky_relu(u_loss, negative_slope=self.uncertainty_skew)

        # critic loss is q loss plus uncertainty loss
        critic_loss = q_loss.mean() + u_loss.mean()

        log = dict()
        log["q_std"] = math.sqrt(self.q_var)
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
        combined_q = self.critic(states, actions_labels)

        # splice the output to get what we want
        q = combined_q[0, 0, ...]
        epistemic = combined_q[1, 1, ...].detach() / self.q_var

        # expectations of Q with clipped double Q
        q, _ = torch.min(q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(q * dones)

        # supervisory loss is difference between predicted and label
        sup_loss = func.mse_loss(labels, actions, reduction="none")
        sup_loss *= self.supervision_lambda

        # entropy calculation
        if self.use_entropy:
            ent_loss = self.log_alpha.exp().detach() * entropies * dones
            ent_loss = ent_loss.mean()
        else:
            ent_loss = 0.0

        # tanh monotonic function
        sup_scale = (
            self.confidence_lambda
            * torch.clamp(epistemic - self.confidence_offset, min=0.0) ** 2
        )
        sup_scale = torch.tanh(sup_scale)

        # convex combo
        rnf_loss = ((1.0 - sup_scale) * rnf_loss).mean()
        sup_loss = (sup_scale * sup_loss).mean()

        # sum the losses
        actor_loss = rnf_loss + sup_loss + ent_loss

        log = dict()
        log["sup_scale"] = sup_scale.mean().detach()
        log["sup_scale_std"] = sup_scale.std().detach()
        log["uncertainty"] = epistemic.mean().detach()

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
