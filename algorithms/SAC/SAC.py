#!/usr/bin/env python3
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as func

from .SACNet import GaussianActor, Q_Ensemble


class SAC(nn.Module):
    """
    Soft Actor Critic
    """

    def __init__(
        self,
        act_size,
        obs_att_size,
        obs_img_size,
        entropy_tuning=True,
        target_entropy=None,
        discount_factor=0.99,
    ):
        super().__init__()

        self.obs_att_size = obs_att_size
        self.obs_img_size = obs_img_size
        self.act_size = act_size
        self.use_entropy = entropy_tuning
        self.gamma = discount_factor

        # actor head
        self.actor = GaussianActor(act_size, obs_att_size, obs_img_size)

        # twin delayed Q networks
        self.critic = Q_Ensemble(act_size, obs_att_size, obs_img_size)
        self.critic_target = Q_Ensemble(act_size, obs_att_size, obs_img_size).eval()

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

    def calc_critic_loss(
        self, obs_att, obs_img, actions, rewards, next_obs_atti, next_obs_targ, terms
    ) -> tuple[torch.FloatTensor, dict]:
        """
        obs_att is of shape B x input_shape
        obs_img is of shape B x C x H x W
        actions is of shape B x act_size
        rewards is of shape B x 1
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # current predicted q
        current_q = self.critic(obs_att, obs_img, actions)

        # compute next q and target_q
        with torch.no_grad():
            # sample the next actions based on the current policy
            output = self.actor(next_obs_atti, next_obs_targ)
            next_actions, log_probs = self.actor.sample(*output)

            # get the next q and f lists and get the value, then...
            next_q = self.critic_target(next_obs_atti, next_obs_targ, next_actions)

            # ...take the min among ensembles
            next_q, _ = torch.min(next_q, dim=-1, keepdim=True)

            # q_target = reward + next_q
            target_q = (
                rewards
                + (-self.log_alpha.exp().detach() * log_probs + self.gamma * next_q)
                * terms
            )

        # calculate bellman loss and take expectation over all networks
        critic_loss = ((current_q - target_q) ** 2).mean()

        # some logging parameters
        log = dict()
        log["target_q"] = target_q.mean().detach()
        log["q_loss"] = critic_loss.mean().detach()

        return critic_loss, log

    def calc_actor_loss(
        self, obs_att, obs_img, terms
    ) -> tuple[torch.FloatTensor, dict]:
        """
        obs_att, obs_img is of shape B x input_shape
        terms is of shape B x 1
        """
        terms = 1.0 - terms

        # We re-sample actions to calculate expectations of Q.
        output = self.actor(obs_att, obs_img)
        actions, entropies = self.actor.sample(*output)

        # expected q for actions
        expected_q = self.critic(obs_att, obs_img, actions)

        """ REINFORCEMENT LOSS """
        # expectations of Q with clipped double Q
        expected_q, _ = torch.min(expected_q, dim=-1, keepdim=True)

        # reinforcement target is maximization of Q * done
        rnf_loss = -(expected_q * terms).mean()

        """ ENTROPY LOSS"""
        # entropy calculation
        if self.use_entropy:
            ent_loss = self.log_alpha.exp().detach() * entropies * terms
            ent_loss = ent_loss.mean()
        else:
            ent_loss = 0.0

        """ TOTAL LOSS DERIVATION"""
        # sum the losses
        actor_loss = rnf_loss + ent_loss

        log = dict()
        log["actor_loss"] = actor_loss.mean().detach()

        return actor_loss, log

    def calc_alpha_loss(self, obs_att, obs_img) -> tuple[torch.Tensor, dict]:
        """
        obs_att, obs_img is of shape B x input_shape
        """
        if not self.entropy_tuning:
            return torch.zeros(1), {}

        output = self.actor(obs_att, obs_img)
        _, log_probs = self.actor.sample(*output)

        # Intuitively, we increse alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -(
            self.log_alpha * (self.target_entropy + log_probs).detach()
        ).mean()

        log = dict()
        log["log_alpha"] = self.log_alpha.item()
        log["mean_entropy"] = -log_probs.mean().detach()

        return entropy_loss, log
