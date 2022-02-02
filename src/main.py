import os
from signal import SIGINT, signal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import wandb
from carracing import Environment
from e2SAC.UASAC import UASAC
from shebangs import check_venv, parse_set, shutdown_handler
from utils.helpers import Helpers, cpuize, gpuize
from utils.replay_buffer import ReplayBuffer


def train(set):

    env = setup_env(set)
    net, net_helper, optim_set, sched_set, optim_helper = setup_nets(set)
    memory = ReplayBuffer(set.buffer_size)
    eval_perf = -100.0
    max_eval_perf = -100.0

    for epoch in range(set.start_epoch, set.epochs):
        """EVAL RUN"""
        if epoch % set.eval_epoch_ratio == 0 and epoch != 0:
            # for logging
            eval_perf = env.evaluate(set, net)
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
                obs, _, _, lbl = env.get_state()

                ent = 0.0
                if epoch < set.exploration_epochs:
                    action = np.random.uniform(-1.0, 1.0, 2)
                else:
                    output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                    action, ent = net.actor.sample(*output)
                    action = cpuize(action)[0]
                    ent = cpuize(ent)[0]

                # get the next state and other stuff
                next_obs, rew, dne, _ = env.step(action)

                # store stuff in mem
                memory.push((obs, action, rew, next_obs, dne, lbl))

                # log progress
                mean_entropy.append(ent)
                frame = np.uint8(obs[:3, ...] * 127.5 + 127.5).transpose(1, 2, 0)
                video_log.append(Image.fromarray(frame))

        # for logging
        total_reward = env.cumulative_reward
        mean_entropy = np.mean(np.array(mean_entropy))
        video_log[0].save(
            "./resource/video_log.gif",
            save_all=True,
            append_images=video_log[1:],
            optimize=False,
            duration=20,
            loop=0,
        )

        """ TRAINING RUN """
        sup_scale_std = 0
        sup_scale_mean = 0

        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=set.batch_size, shuffle=True, drop_last=False
        )

        for i in range(
            int(set.repeats_per_buffer + set.repeats_per_buffer_scale * epoch)
        ):
            for j, stuff in enumerate(dataloader):
                net.train()

                batch = int(set.buffer_size / set.batch_size) * i + j
                states = gpuize(stuff[0], set.device)
                actions = gpuize(stuff[1], set.device)
                rewards = gpuize(stuff[2], set.device)
                next_states = gpuize(stuff[3], set.device)
                dones = gpuize(stuff[4], set.device)
                labels = gpuize(stuff[5], set.device)

                # train critic
                regularizer = None
                for _ in range(set.critic_update_multiplier):
                    net.zero_grad()
                    q_loss, regularizer = net.calc_critic_loss(
                        states, actions, rewards, next_states, dones
                    )
                    q_loss.backward()
                    optim_set["critic"].step()
                    sched_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(set.actor_update_multiplier):
                    net.zero_grad()
                    rnf_loss, sup_loss, sup_scale, reg_loss = net.calc_actor_loss(
                        states, dones, labels
                    )
                    actor_loss = (
                        (set.reg_lambda * regularizer * reg_loss).mean()
                        + ((1.0 - sup_scale) * rnf_loss).mean()
                        + (sup_scale * sup_loss).mean()
                    )
                    actor_loss.backward()
                    optim_set["actor"].step()
                    sched_set["actor"].step()

                    # train entropy regularizer
                    if net.use_entropy:
                        net.zero_grad()
                        ent_loss = net.calc_alpha_loss(states)
                        ent_loss.backward()
                        optim_set["alpha"].step()
                        sched_set["alpha"].step()

                    # for logging
                    sup_scale_mean = sup_scale.mean().item()
                    sup_scale_std = sup_scale.std().item()

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
                        "video": wandb.Video("./resource/video_log.gif"),
                        "epoch": epoch,
                        "total_reward": total_reward,
                        "eval_perf": eval_perf,
                        "max_eval_perf": max_eval_perf,
                        "mean_entropy": mean_entropy,
                        "sup_scale": sup_scale_mean,
                        "sup_scale_std": sup_scale_std,
                        "log_alpha": net.log_alpha.item(),
                        "num_episodes": epoch,
                        "num_transitions": memory.__len__(),
                    }
                    wandb.log(metrics)


def display(set):
    env = setup_env(set)

    net = None
    if True:
        net, _, _, _, _ = setup_nets(set)

    env.display(set, net)


def evaluate(set):
    env = setup_env(set)

    net = None
    if True:
        net, _, _, _, _ = setup_nets(set)

    print(env.evaluate(set, net))


def setup_env(set):
    env = Environment()
    set.num_actions = env.num_actions

    return env


def setup_nets(set):
    net_helper = Helpers(
        mark_number=set.net_number,
        version_number=set.net_version,
        weights_location=set.weights_directory,
        epoch_interval=set.epoch_interval,
        batch_interval=set.batch_interval,
    )
    optim_helper = Helpers(
        mark_number=0,
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
        confidence_cutoff=set.confidence_cutoff,
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
    elif set.evaluate:
        evaluate(set)
    else:
        print("Guess this is life now.")

    """ SCRIPTS END """

    if set.shutdown:
        os.system("poweroff")
