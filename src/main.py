import os
from signal import SIGINT, signal

import numpy as np
import torch
import torch.optim as optim
from PIL import Image

import wandb

# from carracing_dr_env import Environment
from carracing_env import Environment
from e2SAC.UASAC import UASAC
from shebangs import check_venv, parse_set, shutdown_handler
from utils.helpers import Helpers, cpuize, gpuize
from utils.replay_buffer import ReplayBuffer


def train(set):
    env = setup_env(set)
    net, net_helper, optim_set, optim_helper = setup_nets(set)
    memory = ReplayBuffer(set.buffer_size)

    to_log = dict()
    to_log["epoch"] = 0
    to_log["eval_perf"] = -np.inf
    to_log["max_eval_perf"] = -np.inf
    last_eval_step = 0

    while memory.count <= set.total_steps + set.eval_steps_ratio:
        to_log["epoch"] += 1

        """EVAL RUN"""
        if memory.count - last_eval_step > set.eval_steps_ratio:
            last_eval_step += set.eval_steps_ratio
            to_log["eval_perf"] = env.evaluate(set, net)
            to_log["max_eval_perf"] = max(
                [to_log["max_eval_perf"], to_log["eval_perf"]]
            )

        """ENVIRONMENT INTERACTION"""
        env.reset()
        env.train()
        net.eval()
        net.zero_grad()

        with torch.no_grad():
            # video_log = []
            while not env.is_done:
                # get the initial state and label
                obs, _, _, lbl = env.get_state()

                if memory.count < set.exploration_steps:
                    action = env.env.action_space.sample()
                else:
                    output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                    action, _ = net.actor.sample(*output)
                    action = cpuize(action).squeeze(0)

                # get the next state and other stuff
                next_obs, rew, dne, _ = env.step(action)

                # store stuff in mem
                memory.push((obs, action, rew, next_obs, dne, lbl))

                # log progress
                # frame = np.uint8(obs[:3, ...] * 127.5 + 127.5).transpose(1, 2, 0)
                # video_log.append(Image.fromarray(frame))

            # for logging
            to_log["total_reward"] = env.cumulative_reward
            # video_log[0].save(
            #     "./resource/video_log.gif",
            #     save_all=True,
            #     append_images=video_log[1:],
            #     optimize=False,
            #     duration=20,
            #     loop=0,
            # )

        """TRAINING RUN"""
        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=set.batch_size, shuffle=True, drop_last=False
        )

        for repeat_num in range(set.repeats_per_buffer):
            for batch_num, stuff in enumerate(dataloader):
                net.train()

                states = gpuize(stuff[0], set.device)
                actions = gpuize(stuff[1], set.device)
                rewards = gpuize(stuff[2], set.device)
                next_states = gpuize(stuff[3], set.device)
                dones = gpuize(stuff[4], set.device)
                labels = gpuize(stuff[5], set.device)

                # train critic
                for _ in range(set.critic_update_multiplier):
                    net.zero_grad()
                    q_loss, log = net.calc_critic_loss(
                        states, actions, rewards, next_states, dones
                    )
                    to_log = {**to_log, **log}
                    q_loss.backward()
                    optim_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(set.actor_update_multiplier):
                    net.zero_grad()
                    rnf_loss, log = net.calc_actor_loss(states, dones, labels)
                    to_log = {**to_log, **log}
                    rnf_loss.backward()
                    optim_set["actor"].step()

                    # train entropy regularizer
                    if net.use_entropy:
                        net.zero_grad()
                        ent_loss, log = net.calc_alpha_loss(states)
                        to_log = {**to_log, **log}
                        ent_loss.backward()
                        optim_set["alpha"].step()

                """WEIGHTS SAVING"""
                net_weights = net_helper.training_checkpoint(
                    loss=-to_log["eval_perf"], batch=0, epoch=to_log["epoch"]
                )
                net_optim_weights = optim_helper.training_checkpoint(
                    loss=-to_log["eval_perf"], batch=0, epoch=to_log["epoch"]
                )
                if net_weights != -1:
                    torch.save(net.state_dict(), net_weights)
                if net_optim_weights != -1:
                    optim_dict = dict()
                    for key in optim_set:
                        optim_dict[key] = optim_set[key].state_dict()
                    torch.save(
                        {
                            "optim": optim_dict,
                            "lowest_running_loss": optim_helper.lowest_running_loss,
                            "epoch": to_log["epoch"],
                        },
                        net_optim_weights,
                    )

                """WANDB"""
                if set.wandb and repeat_num == 0 and batch_num == 0:
                    to_log["num_transitions"] = memory.count
                    # to_log["video"] = wandb.Video("./resource/video_log.gif")
                    to_log["buffer_size"] = memory.__len__()
                    wandb.log(to_log)


def display(set):
    env = setup_env(set)

    net = None
    if False:
        net, _, _, _ = setup_nets(set)

    env.display(set, net)


def evaluate(set):
    env = setup_env(set)

    net = None
    if False:
        net, _, _, _ = setup_nets(set)

    print(env.evaluate(set, net))


def setup_env(set):
    env = Environment(randomize=(not set.no_randomize))
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
        discount_factor=set.discount_factor,
        confidence_lambda=set.confidence_lambda,
        supervision_lambda=set.supervision_lambda,
        n_var_samples=set.n_var_samples,
    ).to(set.device)
    actor_optim = optim.AdamW(
        net.actor.parameters(), lr=set.learning_rate, amsgrad=True
    )
    critic_optim = optim.AdamW(
        net.critic.parameters(), lr=set.learning_rate, amsgrad=True
    )
    alpha_optim = optim.AdamW([net.log_alpha], lr=0.01, amsgrad=True)

    optim_set = dict()
    optim_set["actor"] = actor_optim
    optim_set["critic"] = critic_optim
    optim_set["alpha"] = alpha_optim

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

        net_helper.lowest_running_loss = checkpoint["lowest_running_loss"]
        optim_helper.lowest_running_loss = checkpoint["lowest_running_loss"]
        print(f"Lowest Running Loss for Net: {net_helper.lowest_running_loss}")

    # torch.save(net.actor.net.state_dict(), "./suboptimal.pth")
    # exit()

    return net, net_helper, optim_set, optim_helper


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
