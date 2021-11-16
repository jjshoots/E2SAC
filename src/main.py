import os
from signal import signal, SIGINT

import cv2
import torch
import wandb
import numpy as np

import torch
import torch.optim as optim

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
    max_mean_reward = -100
    max_eval_perf = -100

    for epoch in range(set.start_epoch, set.epochs):
        # reinitialize buffers
        mean_reward = []
        total_reward = []
        entropy_tracker = []

        states = np.zeros((set.num_envs, envs[0].frame_stack, *envs[0].image_size))
        next_states = np.zeros((set.num_envs, envs[0].frame_stack, *envs[0].image_size))
        actions = np.zeros((set.num_envs, set.num_actions))
        rewards = np.zeros((set.num_envs, 1))
        dones = np.zeros((set.num_envs, 1))
        labels = np.zeros((set.num_envs, set.num_actions))
        next_labels = np.zeros((set.num_envs, set.num_actions))
        entropy = np.zeros((set.num_envs, 1))

        # eval
        for env in envs: env.reset()
        net.eval()
        net.zero_grad()

        eval_perf = []
        while len(eval_perf) < set.eval_num_traj:
            with torch.no_grad():
                # get the initial state and action
                for i, env in enumerate(envs):
                    obs, _, _, _ = env.get_state()
                    states[i] = obs

                output = net.actor(gpuize(states, set.device))
                actions = cpuize(net.actor.infer(*output))

                # get the next state and reward
                for i, env in enumerate(envs):
                    _, _, dne, _ = env.step(actions[i], early_end=False)

                    if dne:
                        eval_perf.append(env.cumulative_reward)
                        env.reset()

        # gather data
        for env in envs: env.reset()
        net.eval()
        net.zero_grad()

        transitions = 0
        while transitions < set.transitions_per_epoch:
            with torch.no_grad():
                # get the initial state and action
                for i, env in enumerate(envs):
                    transitions += 1
                    obs, _, _, lbl = env.get_state()
                    states[i] = obs
                    labels[i] = lbl

                output = net.actor(gpuize(states, set.device))
                o1, ent, _ = net.actor.sample(*output)
                actions = cpuize(o1)
                entropy = cpuize(ent)

                # get the next state and other stuff
                for i, env in enumerate(envs):
                    obs, rew, dne, lbl = env.step(actions[i])
                    next_states[i] = obs
                    rewards[i] = rew
                    dones[i] = dne
                    next_labels[i] = lbl

                    if dne:
                        total_reward.append(env.cumulative_reward)
                        env.reset()
                        num_episodes += 1

                # store stuff in mem
                for stuff in zip(states, actions, rewards, next_states, dones, labels):
                    memory.push(stuff)

                # log progress
                mean_reward.append(np.mean(rewards))
                entropy_tracker.append(np.mean(entropy))

        # for logging
        total_reward = np.mean(np.array(total_reward))
        entropy_tracker = np.mean(np.array(entropy_tracker))
        mean_reward = np.mean(np.array(mean_reward))
        max_mean_reward = max([max_mean_reward, mean_reward])
        eval_perf = np.mean(np.array(eval_perf))
        max_eval_perf = max([max_eval_perf, eval_perf])

        # train on data
        net.train()
        dataloader = torch.utils.data.DataLoader(memory, batch_size=set.batch_size, shuffle=True, drop_last=False)

        for i in range(set.repeats_per_buffer):
            for j, stuff in enumerate(dataloader):
                net.zero_grad()

                batch = int(set.buffer_size / set.batch_size) * i + j
                states = gpuize(stuff[0], set.device)
                actions = gpuize(stuff[1], set.device)
                rewards = gpuize(stuff[2], set.device)
                next_states = gpuize(stuff[3], set.device)
                dones = gpuize(stuff[4], set.device)
                labels = gpuize(stuff[5], set.device)

                # train critic
                q_loss, reg_scale = net.calc_critic_loss(states, actions, rewards, next_states, dones)
                q_loss.backward()
                optim_set['critic'].step()
                sched_set['critic'].step()
                net.update_q_target()

                # train actor
                rnf_loss, sup_loss, sup_scale, reg_loss = net.calc_actor_loss(states, dones, labels)
                actor_loss = set.reg_lambda * (sup_loss / reg_loss).mean().detach() * (reg_scale * reg_loss).mean() \
                             + ((1. - sup_scale) * rnf_loss).mean() + (sup_scale * sup_loss).mean()
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
                net_weights = net_helper.training_checkpoint(loss=-total_reward, batch=batch, epoch=epoch)
                net_optim_weights = optim_helper.training_checkpoint(loss=-total_reward, batch=batch, epoch=epoch)
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
                if set.wandb:
                    metrics = {
                                'epoch': epoch,
                                'mean_reward': mean_reward,
                                'total_reward': total_reward,
                                'max_mean_reward': max_mean_reward,
                                'max_eval_perf': max_eval_perf,
                                'mean_entropy': entropy_tracker,
                                'sup_scale': sup_scale.mean().item(),
                                'log_alpha': net.log_alpha.item(),
                                'num_episodes': num_episodes
                               }
                    wandb.log(metrics)


def display(set):

    use_net = False

    env = setup_envs(set)[0]
    net = None
    if use_net:
        net, _, _, _, _ = setup_nets(set)
        net.eval()

    actions = np.zeros((set.num_envs, set.num_actions))

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)

    while True:
        obs, rwd, dne, lbl = env.step(actions[0])

        if dne:
            actions *= 0.
            env.reset()

        if use_net:
            state = gpuize(obs, set.device).unsqueeze(0)
            output = net.actor(state)
            # actions = cpuize(net.actor.sample(*output)[0])
            actions = cpuize(net.actor.infer(*output))
        else:
            actions[0] = lbl

        display = np.concatenate([*obs], 1)
        display = np.uint8((display + 1) / 2 * 255)
        cv2.imshow('display', display)
        cv2.waitKey(int(1000 / 24))


def setup_envs(set):
    envs = [Environment() for _ in range(set.num_envs)]
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
        confidence_scale=set.confidence_scale,
        confidence_cutoff=set.confidence_cutoff
    ).to(set.device)
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

    sched_set = dict()
    sched_set['actor'] = actor_sched
    sched_set['critic'] = critic_sched
    sched_set['alpha'] = alpha_sched

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
        # set.start_epoch = checkpoint['epoch']
        print(f'Lowest Running Loss for Net: {net_helper.lowest_running_loss} @ epoch {set.start_epoch}')

    return \
        net, net_helper, optim_set, sched_set, optim_helper


if __name__ == '__main__':
    signal(SIGINT, shutdown_handler)
    set = parse_set()
    torch.autograd.set_detect_anomaly(True)

    """ SCRIPTS HERE """

    if set.display:
        display(set)
    elif set.train:
        train(set)
    else:
        print('Guess this is life now.')

    """ SCRIPTS END """

    if set.shutdown:
        os.system('poweroff')
