#!/usr/bin/env python3
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as func

from .CCGENet import GaussianActor, Q_Ensemble


class CCGE(nn.Module):
    """
    Critic Confidence Guided Exploration
    """

    def __init__(
        self,
        act_size,
        obs_size,
        entropy_tuning=True,
        target_entropy=None,
        discount_factor=0.99,
        confidence_lambda=1.0,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.act_size = act_size
        self.use_entropy = entropy_tuning
        self.gamma = discount_factor
        self.confidence_lambda = confidence_lambda

        # actor head
        self.actor = GaussianActor(act_size, obs_size)

        # twin delayed Q networks
        self.critic = Q_Ensemble(act_size, obs_size)
        self.critic_target = Q_Ensemble(act_size, obs_size).eval()

        # copy weights and disable gradients for the target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # tune entropy using log alpha, starts with 0
        self.entropy_tuning = entropy_tuning
        if entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -float(act_size)
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

    def update_q_target(self, tau=0.02):
        # polyak averaging update for target q network
        for target, source in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1.0 - tau) + source.data * tau)

    def calc_sup_scale(
        self, obs, actions, labels
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, dict]:
        # put all actions and labels and states through critic
        # output of each is B x num_networks
        a_out= self.critic(obs, actions)
        act_q, act_u = a_out[0], a_out[1]
        l_out = self.critic(obs, labels)
        lbl_q, lbl_u = l_out[0], l_out[1]

        """ SUPERVISION SCALE DERIVATION """
        # uncertainty is upper bound difference between suboptimal and learned
        uncertainty = (
            (
                lbl_q.mean(dim=-1, keepdim=True)
                + lbl_u.max(dim=-1, keepdim=True)[0]
            )
            - (
                act_q.mean(dim=-1, keepdim=True)
                + act_u.min(dim=-1, keepdim=True)[0]
            )
        ).detach()

        # normalize uncertainty
        uncertainty = (
            uncertainty / act_q.mean(dim=-1, keepdim=True).abs()
        ).detach()

        # supervision scale is a switch
        sup_scale = (uncertainty > self.confidence_lambda) * 1.0

        log = dict()
        log["uncertainty"] = uncertainty.mean().detach()

        return sup_scale, act_q, log

    def calc_critic_loss(
        self, obs, actions, rewards, next_obs, terms
    ) -> tuple[torch.FloatTensor, dict]:
        """
        obs is of shape B x C x H x W
        actions is of shape B x act_size
        rewards is of shape B x 1
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # current predicted f and f
        current_q, current_f = self.critic(obs, actions)

        # compute next q and next f and target_q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_obs)
            next_actions, log_probs = self.actor.sample(*output)

            # get the next q and f lists and get the value, then...
            next_q, next_f = self.critic_target(
                next_obs, next_actions
            )

            # ...take the min among ensembles
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)
            next_f, _ = torch.min(next_f, dim=-1, keepdim=True)

            # q_target = reward + next_q
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * log_probs + self.gamma * next_q)
                * terms
            )

        # calculate bellman loss and take expectation over all networks
        bellman_loss = (current_q - target_q) ** 2
        bellman_loss = bellman_loss.mean(dim=-1, keepdim=True)

        # q loss is just bellman error
        q_loss = bellman_loss.mean()

        # f_target = projected_bellman_error + next_f
        # U_target = sqrt(bellman_error + next_f^2)
        target_f = (bellman_loss.detach() + (self.gamma * next_f * terms) ** 2).sqrt()
        f_loss = ((current_f - target_f) ** 2).mean()

        # critic loss is q loss plus uncertainty loss
        critic_loss = q_loss + f_loss

        # some logging parameters
        log = dict()
        log["target_f"] = target_f.mean().detach()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = q_loss.mean().detach()
        log["f_loss"] = f_loss.mean().detach()
        log["q_td_ratio"] = abs(bellman_loss / target_q).mean().detach()
        log["f_td_ratio"] = abs(target_f / target_q).mean().detach()

        return critic_loss, log

    def calc_actor_loss(
        self, obs, terms, labels
    ) -> tuple[torch.FloatTensor, dict]:
        """
        obs is of shape B x input_shape
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(obs)
        actions, entropies = self.actor.sample(*output)

        # compute supervision scale and expected q for actions
        sup_scale, expected_q, log2 = self.calc_sup_scale(obs, actions, labels)

        """ REINFORCEMENT LOSS """
        # expectations of Q with clipped double Q
        expected_q, _ = torch.min(expected_q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(expected_q * terms)

        """ SUPERVISION LOSS"""
        # supervisory loss is difference between predicted and label
        sup_loss = func.mse_loss(labels, actions, reduction="none")

        """ ENTROPY LOSS"""
        # entropy calculation
        if self.use_entropy:
            ent_loss = self.log_alpha.exp().detach() * entropies * terms
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
        log = {**log, **log2}

        return actor_loss, log

    def calc_alpha_loss(self, obs) -> tuple[torch.FloatTensor, dict]:
        """
        obs is of shape B x input_shape
        """
        if not self.entropy_tuning:
            return torch.zeros(1), {}

        output = self.actor(obs)
        _, log_probs = self.actor.sample(*output)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -(
            self.log_alpha * (self.target_entropy + log_probs).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self.log_alpha.item()
        log["mean_entropy"] = -log_probs.mean().detach()

        return entropy_loss, log
