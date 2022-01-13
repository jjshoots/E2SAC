import os
import sys
from signal import signal, SIGINT

import cv2
import torch
import wandb
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from utility.shebangs import *

from carracing import *

from ai_lib.replay_buffer import *
from ai_lib.SAC import SAC


def train(set):
    env = setup_env(set)
    net, net_helper, optim_set, sched_set, optim_helper = setup_nets(set)
    memory = ReplayBuffer(set.buffer_size)
    eval_perf = 0.0
    max_eval_perf = -100.0

    for epoch in range(set.start_epoch, set.epochs):
        """EVAL RUN"""
        if epoch % set.eval_epoch_ratio == 0:
            env.reset()
            env.eval()
            net.eval()
            net.zero_grad()

            eval_perf = []
            while len(eval_perf) < set.eval_num_traj:
                with torch.no_grad():
                    # get the initial state and action
                    obs, _, _, _ = env.get_state()

                    output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                    action = net.actor.infer(*output)
                    action = cpuize(action)[0]

                    # get the next state and reward
                    _, _, _, _ = env.step(action)

                    if env.is_done:
                        eval_perf.append(env.cumulative_reward)
                        env.reset()

            # for logging
            eval_perf = np.mean(np.array(eval_perf))
            max_eval_perf = max([max_eval_perf, eval_perf])

        """ENVIRONMENT INTERACTION """
        total_reward = []
        mean_entropy = []
        video_log = []

        env.reset()
        env.train()
        net.eval()
        net.zero_grad()

        with torch.no_grad():
            while not env.is_done:
                # get the initial state and label
                obs, _, _, _ = env.get_state()

                # pass states to actor and get actions
                output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                if epoch < set.exploration_epochs:
                    action = np.random.uniform(-1.0, 1.0, 2)
                    ent = 0.0
                else:
                    action, ent = net.actor.sample(*output)
                    action = cpuize(action)[0]
                    ent = cpuize(ent)[0]

                # get the next state and other stuff
                next_obs, rew, dne, _ = env.step(action)

                # store stuff in mem
                memory.push((obs, action, rew, next_obs, dne))

                # log progress
                mean_entropy.append(ent)
                video_log.append(np.uint8(obs[:3, ...] * 127.5 + 127.5))

        # for logging
        total_reward = env.cumulative_reward
        mean_entropy = np.mean(np.array(mean_entropy))
        video_log = np.stack(video_log, axis=0)

        """ TRAINING RUN """
        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=set.batch_size, shuffle=True, drop_last=False
        )

        for i in range(set.repeats_per_buffer):
            for j, stuff in enumerate(dataloader):
                net.train()

                batch = int(set.buffer_size / set.batch_size) * i + j
                states = gpuize(stuff[0], set.device)
                actions = gpuize(stuff[1], set.device)
                rewards = gpuize(stuff[2], set.device)
                next_states = gpuize(stuff[3], set.device)
                dones = gpuize(stuff[4], set.device)

                # train critic
                for _ in range(set.critic_update_multiplier):
                    net.zero_grad()
                    q_loss = net.calc_critic_loss(
                        states, actions, rewards, next_states, dones
                    )
                    q_loss.backward()
                    # nn.utils.clip_grad_norm_(net.critic.parameters(), max_norm=2.0, norm_type=2)
                    optim_set["critic"].step()
                    sched_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(set.actor_update_multiplier):
                    net.zero_grad()
                    rnf_loss = net.calc_actor_loss(states, dones)
                    rnf_loss.backward()
                    # nn.utils.clip_grad_norm_(net.actor.parameters(), max_norm=2.0, norm_type=2)
                    optim_set["actor"].step()
                    sched_set["actor"].step()

                    # train entropy regularizer
                    if net.use_entropy:
                        net.zero_grad()
                        ent_loss = net.calc_alpha_loss(states)
                        ent_loss.backward()
                        optim_set["alpha"].step()
                        sched_set["alpha"].step()

                """ WEIGHTS SAVING """
                net_weights = net_helper.training_checkpoint(
                    loss=-eval_perf, batch=batch, epoch=epoch
                )
                net_optim_weights = optim_helper.training_checkpoint(
                    loss=-eval_perf, batch=batch, epoch=epoch
                )
                if net_weights != -1:
                    torch.save(net.state_dict(), net_weights)
                if net_optim_weights != -1:
                    optim_dict = dict()
                    for key in optim_set:
                        optim_dict[key] = optim_set[key].state_dict()
                    sched_dict = dict()
                    for key in optim_set:
                        sched_dict[key] = sched_set[key].state_dict()
                    torch.save(
                        {
                            "optim": optim_dict,
                            "sched": sched_dict,
                            "lowest_running_loss": optim_helper.lowest_running_loss,
                            "epoch": epoch,
                        },
                        net_optim_weights,
                    )

                """ WANDB """
                if set.wandb and i == 0 and j == 0:
                    metrics = {
                        "video": wandb.Video(video_log, fps=50, format="gif"),
                        "epoch": epoch,
                        "total_reward": total_reward,
                        "eval_perf": eval_perf,
                        "max_eval_perf": max_eval_perf,
                        "mean_entropy": mean_entropy,
                        "log_alpha": net.log_alpha.item(),
                        "num_episodes": epoch,
                    }
                    wandb.log(metrics)


def display(set):

    use_net = True

    env = setup_env(set)
    env.eval()
    net = None
    if use_net:
        net, _, _, _, _ = setup_nets(set)
        net.eval()

    action = np.zeros((set.num_actions))

    cv2.namedWindow("display", cv2.WINDOW_NORMAL)

    while True:
        obs, rwd, dne, lbl = env.step(action)

        if env.is_done:
            print(f"Total Reward: {env.cumulative_reward}")
            env.reset()
            action *= 0.0

        if use_net:
            output = net.actor(gpuize(obs, set.device).unsqueeze(0))
            # action = cpuize(net.actor.sample(*output)[0][0])
            action = cpuize(net.actor.infer(*output))[0]

            # print(action)
            print(
                net.critic.forward(
                    gpuize(obs, set.device).unsqueeze(0), net.actor.infer(*output)[0]
                )[0]
                .squeeze()
                .item()
            )
        else:
            action = lbl[0]

        display = obs[:3, ...]
        display = np.uint8((display * 127.5 + 127.5))
        display = np.transpose(display, (1, 2, 0))
        cv2.imshow("display", display)
        cv2.waitKey(int(1000 / 15))


def setup_env(set):
    env = Environment()
    set.num_actions = env.num_actions

    return env


def setup_nets(set):
    net_helper = Logger(
        mark_number=set.net_number,
        version_number=set.net_version,
        weights_location=set.weights_directory,
        epoch_interval=set.epoch_interval,
        batch_interval=set.batch_interval,
    )
    optim_helper = Logger(
        mark_number=0,
        version_number=set.net_version,
        weights_location=set.optim_weights_directory,
        epoch_interval=set.epoch_interval,
        batch_interval=set.batch_interval,
        increment=False,
    )

    # set up networks and optimizers
    net = SAC(
        num_actions=set.num_actions,
        entropy_tuning=set.use_entropy,
        target_entropy=set.target_entropy,
    ).to(set.device)
    actor_optim = optim.AdamW(net.actor.parameters(), lr=set.starting_LR, amsgrad=True)
    actor_sched = optim.lr_scheduler.StepLR(
        actor_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma
    )
    critic_optim = optim.AdamW(
        net.critic.parameters(), lr=set.starting_LR, amsgrad=True
    )
    critic_sched = optim.lr_scheduler.StepLR(
        critic_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma
    )
    alpha_optim = optim.AdamW([net.log_alpha], lr=set.starting_LR, amsgrad=True)
    alpha_sched = optim.lr_scheduler.StepLR(
        alpha_optim, step_size=set.step_sched_num, gamma=set.scheduler_gamma
    )

    optim_set = dict()
    optim_set["actor"] = actor_optim
    optim_set["critic"] = critic_optim
    optim_set["alpha"] = alpha_optim

    sched_set = dict()
    sched_set["actor"] = actor_sched
    sched_set["critic"] = critic_sched
    sched_set["alpha"] = alpha_sched

    # get latest weight files
    net_weights = net_helper.get_weight_file()
    if net_weights != -1:
        net.load_state_dict(torch.load(net_weights))

    # get latest optimizer states
    net_optimizer_weights = optim_helper.get_weight_file()
    if net_optimizer_weights != -1:
        checkpoint = torch.load(net_optimizer_weights)

        for opt_key in optim_set:
            optim_set[opt_key].load_state_dict(checkpoint["optim"][opt_key])
        for sch_key in sched_set:
            sched_set[sch_key].load_state_dict(checkpoint["optim"][sch_key])

        net_helper.lowest_running_loss = checkpoint["lowest_running_loss"]
        optim_helper.lowest_running_loss = checkpoint["lowest_running_loss"]
        # set.start_epoch = checkpoint['epoch']
        print(
            f"Lowest Running Loss for Net: {net_helper.lowest_running_loss} @ epoch {set.start_epoch}"
        )

    return net, net_helper, optim_set, sched_set, optim_helper


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    set = parse_set()
    check_venv()

    """ SCRIPTS HERE """

    if set.display:
        display(set)
    elif set.train:
        train(set)
    else:
        print("Guess this is life now.")

    """ SCRIPTS END """

    if set.shutdown:
        os.system("poweroff")
