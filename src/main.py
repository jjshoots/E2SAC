import os
import argparse
from sys import exit
from signal import signal, SIGINT

import cv2
import yaml
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from utility.shebangs import *

from carracing import *

from ai_lib.replay_buffer import *
from ai_lib.normal_inverse_gamma import *
from ai_lib.UASAC import UASAC


def train(set):
    envs = setup_envs(set)
    net, net_helper, optim_set, sched_set, optim_helper = setup_nets(set)
    memory = ReplayBuffer(set.buffer_size)
    num_episodes = set.num_envs

    for epoch in range(set.start_epoch, set.epochs):
        # gather the data
        net.eval()
        rewards_tracker = []
        entropy_tracker = []

        states = np.zeros((set.num_envs, envs[0].state_size))
        next_states = np.zeros((set.num_envs, envs[0].state_size))
        actions = np.zeros((set.num_envs, set.num_actions))
        next_actions = np.zeros((set.num_envs, set.num_actions))
        rewards = np.zeros((set.num_envs, 1))
        dones = np.zeros((set.num_envs, 1))
        labels = np.zeros((set.num_envs, set.num_actions))
        next_labels = np.zeros((set.num_envs, set.num_actions))
        entropy = np.zeros((set.num_envs, 1))

        for _ in range(int(set.transitions_per_epoch / set.num_envs)):
            net.zero_grad()

            # get the initial state and action
            for i, env in enumerate(envs):
                obs, _, _, lbl = env.get_state()
                states[i] = obs
                labels[i] = lbl

            output = net.backbone(gpuize(states, set.device))
            output = net.actor(output)
            o1, ent, _ = net.actor.sample(*output)
            actions = cpuize(o1)
            entropy = cpuize(ent)

            # get the next state, next action, and other stuff
            for i, env in enumerate(envs):
                obs, rew, dne, lbl = env.step(actions[i])
                next_states[i] = obs
                rewards[i] = rew
                dones[i] = dne
                next_labels[i] = lbl

                if dne:
                    env.reset()
                    num_episodes += 1

            output = net.backbone(gpuize(states, set.device))
            output = net.actor(output)
            o1, _, _ = net.actor.sample(*output)
            next_actions = cpuize(o1)

            # store stuff in mem
            for stuff in zip(states, next_states,
                             actions, next_actions,
                             rewards, dones, labels):
                memory.push(stuff)

            # log progress
            rewards_tracker.append(np.mean(rewards))
            entropy_tracker.append(np.mean(entropy))

        # for logging
        rewards_tracker = np.mean(np.array(rewards_tracker))
        entropy_tracker = np.mean(np.array(entropy_tracker))

        # train on data
        net.train()
        dataloader = torch.utils.data.DataLoader(memory, batch_size=set.batch_size, shuffle=True, drop_last=False)

        for i in range(set.repeats_per_buffer):
            for j, stuff in enumerate(dataloader):
                net.zero_grad()

                batch = int(set.buffer_size / set.batch_size) * i + j
                states = gpuize(stuff[0], set.device)
                next_states = gpuize(stuff[1], set.device)
                actions = gpuize(stuff[2], set.device)
                next_actions = gpuize(stuff[3], set.device)
                rewards = gpuize(stuff[4], set.device)
                dones = gpuize(stuff[5], set.device)
                labels = gpuize(stuff[6], set.device)

                # train critic
                q_loss, reg_scale = net.calc_critic_loss(states, next_states, actions, next_actions, rewards, dones)
                q_loss.backward()
                optim_set['critic'].step()
                sched_set['critic'].step()
                optim_set['backbone'].step()
                sched_set['backbone'].step()
                net.update_q_target()

                # train actor
                rnf_loss, sup_loss, sup_scale, reg_loss = net.calc_actor_loss(states, dones, labels)
                actor_loss = set.reg_lambda * (sup_loss / reg_loss).mean().detach() * (reg_scale * reg_loss).mean()
                if batch % set.ac_update_ratio == 0:
                    actor_loss = actor_loss + \
                    ((1. - sup_scale) * rnf_loss).mean() + \
                    (sup_scale * sup_loss).mean()
                actor_loss.backward()
                optim_set['actor'].step()
                sched_set['actor'].step()

                # train entropy regularizer
                if net.use_entropy:
                    ent_loss = net.calc_alpha_loss(states)
                    ent_loss.backward()
                    optim_set['alpha'].step()
                    sched_set['alpha'].step()

                # detect whether we need to save the weights file and record the losses
                net_weights = net_helper.training_checkpoint(loss=-rewards_tracker, batch=batch, epoch=epoch)
                net_optim_weights = optim_helper.training_checkpoint(loss=-rewards_tracker, batch=batch, epoch=epoch)
                if net_weights != -1: torch.save(net.state_dict(), net_weights)
                if net_optim_weights != -1:
                    optim_dict = dict()
                    for key in optim_set:
                        optim_dict[key] = optim_set[key].state_dict()
                    sched_dict = dict()
                    for key in optim_set:
                        sched_dict[key] = sched_set[key].state_dict()
                    torch.save({ \
                                'optim': optim_dict,
                                'sched': sched_dict,
                                'lowest_running_loss': optim_helper.lowest_running_loss,
                                'epoch': epoch,
                               },
                              net_optim_weights)

                # wandb
                metrics = { \
                            'epoch': epoch, \
                            'mean_reward': rewards_tracker, \
                            'mean_entropy': entropy_tracker, \
                            'sup_scale': sup_scale.mean().item(), \
                            'log_alpha': net.log_alpha.item(), \
                            'num_episodes': num_episodes
                           } \

                if set.wandb:
                    wandb.log(metrics)


def display(set):
    env = setup_envs(set)[0]
    net, _, _, _, _ = setup_nets(set)
    net.eval()

    actions = np.zeros((set.num_envs, set.num_actions))

    while True:
        obs, rwd, dne, lbl = env.step(actions[0])

        if dne:
            actions *= 0.
            env.reset()

        if True:
            obs = gpuize(obs, set.device).unsqueeze(0)
            output = net.backbone(obs)
            output = net.actor(output)
            actions = cpuize(net.actor.sample(*output)[0])
            # actions = cpuize(net.actor.infer(*output))
        else:
            actions[0] = lbl

        time.sleep(1./24.)


def setup_envs(set):
    display = True if set.num_envs == 1 else False
    envs = [Environment(display=display) for _ in range(set.num_envs)]
    set.num_actions = envs[0].num_actions

    return envs


def setup_nets(set):
    net_helper = Logger(mark_number=set.net_number,
                         version_number=set.net_version,
                         weights_location=set.weights_directory,
                         epoch_interval=set.epoch_interval,
                         batch_interval=set.batch_interval,
                         )
    optim_helper = Logger(mark_number=0,
                           version_number=set.net_version,
                           weights_location=set.optim_weights_directory,
                           epoch_interval=set.epoch_interval,
                           batch_interval=set.batch_interval,
                           increment=False,
                           )

    # set up networks and optimizers
    net = UASAC(
        num_actions=set.num_actions,
        entropy_tuning=set.use_entropy,
        target_entropy=set.target_entropy,
        confidence_scale=set.confidence_scale
    ).to(set.device)
    backbone_optim = optim.AdamW(net.backbone.parameters(), lr=set.starting_LR, amsgrad=True)
    backbone_sched = optim.lr_scheduler.StepLR(backbone_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma)
    actor_optim = optim.AdamW(net.actor.parameters(), lr=set.starting_LR, amsgrad=True)
    actor_sched = optim.lr_scheduler.StepLR(actor_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma)
    critic_optim = optim.AdamW(net.critic.parameters(), lr=set.starting_LR, amsgrad=True)
    critic_sched = optim.lr_scheduler.StepLR(critic_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma)
    alpha_optim = optim.AdamW([net.log_alpha], lr=set.starting_LR, amsgrad=True)
    alpha_sched = optim.lr_scheduler.StepLR(alpha_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma)

    optim_set = dict()
    optim_set['actor'] = actor_optim
    optim_set['critic'] = critic_optim
    optim_set['alpha'] = alpha_optim
    optim_set['backbone'] = backbone_optim

    sched_set = dict()
    sched_set['actor'] = actor_sched
    sched_set['critic'] = critic_sched
    sched_set['alpha'] = alpha_sched
    sched_set['backbone'] = backbone_sched

    # get latest weight files
    net_weights = net_helper.get_weight_file()
    if net_weights != -1: net.load_state_dict(torch.load(net_weights))

    # get latest optimizer states
    net_optimizer_weights = optim_helper.get_weight_file()
    if net_optimizer_weights != -1:
        checkpoint = torch.load(net_optimizer_weights)

        for opt_key in optim_set:
            optim_set[opt_key].load_state_dict(checkpoint['optim'][opt_key])
        for sch_key in sched_set:
            sched_set[sch_key].load_state_dict(checkpoint['optim'][sch_key])

        net_helper.lowest_running_loss = checkpoint['lowest_running_loss']
        optim_helper.lowest_running_loss = checkpoint['lowest_running_loss']
        set.start_epoch = checkpoint['epoch']
        print(f'Lowest Running Loss for Net: {net_helper.lowest_running_loss} @ epoch {set.start_epoch}')

    return \
        net, net_helper, optim_set, sched_set, optim_helper


if __name__ == '__main__':
    signal(SIGINT, shutdown_handler)
    set, args = parse_set_args()
    torch.autograd.set_detect_anomaly(True)

    """ SCRIPTS HERE """

    if args.display:
        display(set)
    elif args.train:
        train(set)
    else:
        print('Guess this is life now.')

    """ SCRIPTS END """

    if args.shutdown:
        os.system('poweroff')
