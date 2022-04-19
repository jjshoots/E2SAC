import os
from signal import SIGINT, signal

import numpy as np
import torch
import torch.optim as optim
from PIL import Image

import wandb
from ESDDQN.ESDDQN import ESDDQN
from discrete_env import Environment
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

        """SWITCH UP ENV"""
        if to_log["epoch"] % set.switchup_epoch == 0:
            # env.switchup()
            # memory.refresh()
            pass

        """ENVIRONMENT INTERACTION"""
        env.reset()
        env.train()
        net.eval()
        net.zero_grad()

        with torch.no_grad():
            video_log = []
            cumulative_uncertainty = []
            while not env.is_done:
                # get the initial state and label
                obs, _, _, _ = env.get_state()

                if memory.count < set.exploration_steps:
                    action = np.random.uniform(-1.0, 1.0, 2)
                else:
                    q, uncertainty = net(gpuize(obs, set.device).unsqueeze(0))
                    action = cpuize(net.sample(q, uncertainty))

                # for logging
                cumulative_uncertainty.append(cpuize(uncertainty).mean())

                # get the next state and other stuff
                next_obs, rew, dne, _ = env.step(action)

                # store stuff in mem
                memory.push((obs, action, rew, next_obs, dne))

                # log progress
                frame = np.uint8(obs[:3, ...] * 127.5 + 127.5).transpose(1, 2, 0)
                video_log.append(Image.fromarray(frame))

            # for logging
            to_log["total_reward"] = env.cumulative_reward
            to_log["runtime_uncertainty"] = np.mean(np.stack(cumulative_uncertainty))
            video_log[0].save(
                "./resource/video_log.gif",
                save_all=True,
                append_images=video_log[1:],
                optimize=False,
                duration=20,
                loop=0,
            )

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

                # train
                net.zero_grad()
                loss, log = net.calc_loss(states, actions, rewards, next_states, dones)
                to_log = {**to_log, **log}
                loss.backward()
                optim_set["ddqn"].step()
                net.update_q_target()

                """ WEIGHTS SAVING """
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
                    to_log["video"] = wandb.Video("./resource/video_log.gif")
                    to_log["buffer_size"] = memory.__len__()
                    wandb.log(to_log)


def display(set):
    env = setup_env(set)

    net = None
    if True:
        net, _, _, _ = setup_nets(set)

    env.display(set, net)


def evaluate(set):
    env = setup_env(set)

    net = None
    if False:
        net, _, _, _ = setup_nets(set)

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
    net = ESDDQN(
        num_actions=set.num_actions,
        exploration_epsilon=set.exploration_epsilon,
    ).to(set.device)
    ddqn_optim = optim.AdamW(net.parameters(), lr=set.learning_rate, amsgrad=True)

    optim_set = dict()
    optim_set["ddqn"] = ddqn_optim

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
