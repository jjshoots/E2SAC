from signal import SIGINT, signal

import numpy as np
import torch
import torch.optim as optim

import wandb
from e2SAC.UASAC import UASAC
from dm_control_env import Environment
from wingman import Wingman, cpuize, gpuize, shutdown_handler, ReplayBuffer


def train(wm: Wingman):
    # grab config
    cfg = wm.cfg

    # setup env, model, replaybuffer
    env = setup_env(wm)
    net, optim_set = setup_nets(wm)
    memory = ReplayBuffer(cfg.buffer_size)

    to_log = dict()
    to_log["epoch"] = 0
    to_log["eval_perf"] = -np.inf
    to_log["max_eval_perf"] = -np.inf
    last_eval_step = 0

    while memory.count <= cfg.total_steps:
        to_log["epoch"] += 1

        """EVAL RUN"""
        if memory.count - last_eval_step > cfg.eval_steps_ratio:
            last_eval_step += cfg.eval_steps_ratio
            to_log["eval_perf"] = env.evaluate(cfg, net)
            to_log["max_eval_perf"] = max(
                [to_log["max_eval_perf"], to_log["eval_perf"]]
            )

        """ENVIRONMENT INTERACTION"""
        env.reset()
        env.train()
        net.eval()
        net.zero_grad()

        with torch.no_grad():
            while not env.is_done:
                # get the initial state and label
                obs, _, _, lbl = env.get_state()

                if memory.count < cfg.exploration_steps:
                    action = env.env.action_space.sample()
                else:
                    output = net.actor(gpuize(obs, cfg.device).unsqueeze(0))
                    action, _ = net.actor.sample(*output)
                    action = cpuize(action).squeeze(0)

                # get the next state and other stuff
                next_obs, rew, dne, _ = env.step(action)

                # store stuff in mem
                memory.push((obs, action, rew, next_obs, dne, lbl))

            # for logging
            to_log["total_reward"] = env.cumulative_reward

        """TRAINING RUN"""
        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )

        for repeat_num in range(int(cfg.repeats_per_buffer)):
            for batch_num, stuff in enumerate(dataloader):
                net.train()

                states = gpuize(stuff[0], cfg.device)
                actions = gpuize(stuff[1], cfg.device)
                rewards = gpuize(stuff[2], cfg.device)
                next_states = gpuize(stuff[3], cfg.device)
                dones = gpuize(stuff[4], cfg.device)
                labels = gpuize(stuff[5], cfg.device)

                # train critic
                for _ in range(cfg.critic_update_multiplier):
                    net.zero_grad()
                    q_loss, log = net.calc_critic_loss(
                        states, actions, rewards, next_states, dones
                    )
                    to_log = {**to_log, **log}
                    q_loss.backward()
                    optim_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(cfg.actor_update_multiplier):
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
                if cfg.wandb and repeat_num == 0 and batch_num == 0:
                    to_log["num_transitions"] = memory.count
                    to_log["buffer_size"] = memory.__len__()
                    wandb.log(to_log)


def display(wm: Wingman):
    cfg = wm.cfg
    env = setup_env(wm)

    net = None
    if True:
        net, _, _, _ = setup_nets(cfg)

    env.display(cfg, net)


def evaluate(wm: Wingman):
    cfg = wm.cfg
    env = setup_env(wm)

    net = None
    if False:
        net, _, _, _ = setup_nets(cfg)

    print(env.evaluate(cfg, net))


def setup_env(wm: Wingman):
    cfg = wm.cfg
    env = Environment(cfg.env_name, sub_size=cfg.sub_size)
    cfg.num_actions = env.num_actions
    cfg.state_size = env.state_size

    return env


def setup_nets(wm: Wingman):
    cfg = wm.cfg

    # set up networks and optimizers
    net = UASAC(
        num_actions=cfg.num_actions,
        state_size=cfg.state_size,
        entropy_tuning=cfg.use_entropy,
        target_entropy=cfg.target_entropy,
        discount_factor=cfg.discount_factor,
        confidence_lambda=cfg.confidence_lambda,
        supervision_lambda=cfg.supervision_lambda,
        n_var_samples=cfg.n_var_samples,
    ).to(cfg.device)
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
    has_weights, model_file, optim_file = wm.get_weight_files()
    if has_weights:
        # load the model
        net.load_state_dict(torch.load(model_file))

        # load the optimizer
        checkpoint = torch.load(model_file)

        for opt_key in optim_set:
            optim_set[opt_key].load_state_dict(checkpoint["optim"][opt_key])

        print(f"Lowest Running Loss for Net: {wm.lowest_loss}")

    # torch.save(net.actor.net.state_dict(), f"./{set.env_name}_big.pth")
    # exit()

    return net, optim_set


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./settings.yaml")

    """ SCRIPTS HERE """

    if wm.cfg.display:
        display(wm)
    elif wm.cfg.train:
        train(wm)
    elif wm.cfg.evaluate:
        evaluate(wm)
    else:
        print("Guess this is life now.")
