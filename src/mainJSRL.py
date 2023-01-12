import math
from signal import SIGINT, signal

import numpy as np
import torch
import torch.optim as optim
from wingman import ReplayBuffer, Wingman, cpuize, gpuize, shutdown_handler

from algorithms import SAC
from d4rl_env import Environment


def train(wm: Wingman):
    # grab config
    cfg = wm.cfg

    # setup env, model, replaybuffer
    env = setup_env(wm)
    net, optim_set = setup_nets(wm)
    memory = ReplayBuffer(cfg.buffer_size)

    wm.log["epoch"] = 0
    wm.log["eval_perf"] = -math.inf
    wm.log["max_eval_perf"] = -math.inf
    next_eval_step = 0

    while memory.count <= cfg.total_steps:
        wm.log["epoch"] += 1

        """EVAL RUN"""
        if memory.count >= next_eval_step:
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio
            wm.log["success_rate"], wm.log["eval_perf"] = env.evaluate(cfg, net)
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )

        """ENVIRONMENT INTERACTION"""
        env.reset()
        net.eval()
        net.zero_grad()

        with torch.no_grad():

            # JSRL steps
            steps = 0
            oracle_steps = np.random.randint(int(200 * 0.99))

            while not env.ended:
                # get the initial state
                obs = env.state

                # JSRL if needed
                if steps < oracle_steps:
                    _ = env.step(env.get_label(obs))
                else:
                    # get the action from policy
                    output = net.actor(gpuize(obs, cfg.device).unsqueeze(0))
                    act, _ = net.actor.sample(*output)
                    act = cpuize(act)

                    # get the next state and other stuff
                    next_obs, rew, term = env.step(act)

                    # store stuff in mem
                    memory.push(
                        [
                            obs,
                            act,
                            rew,
                            next_obs,
                            term,
                        ]
                    )

                # increment our step counter
                steps += 1

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
                terms = gpuize(stuff[4], cfg.device)

                # train critic
                for _ in range(cfg.critic_update_multiplier):
                    net.zero_grad()
                    q_loss, log = net.calc_critic_loss(
                        states,
                        actions,
                        rewards,
                        next_states,
                        terms,
                    )
                    wm.log = {**wm.log, **log}
                    q_loss.backward()
                    optim_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(cfg.actor_update_multiplier):
                    net.zero_grad()
                    rnf_loss, log = net.calc_actor_loss(states, terms)
                    wm.log = {**wm.log, **log}
                    rnf_loss.backward()
                    optim_set["actor"].step()

                    # train entropy regularizer
                    if net.use_entropy:
                        net.zero_grad()
                        ent_loss, log = net.calc_alpha_loss(states)
                        wm.log = {**wm.log, **log}
                        ent_loss.backward()
                        optim_set["alpha"].step()

                """WANDB"""
                wm.log["num_transitions"] = memory.count
                wm.log["buffer_size"] = memory.__len__()

                """WEIGHTS SAVING"""
                to_update, model_file, optim_file = wm.checkpoint(
                    loss=-float(wm.log["eval_perf"]), step=wm.log["num_transitions"]
                )
                if to_update:
                    torch.save(net.state_dict(), model_file)

                    optim_dict = dict()
                    for key in optim_set:
                        optim_dict[key] = optim_set[key].state_dict()
                    torch.save(optim_dict, optim_file)


def eval_display(wm: Wingman):
    cfg = wm.cfg
    env = setup_env(wm)

    if not cfg.debug:
        model, _ = setup_nets(wm)
    else:
        model = None

    if wm.cfg.display:
        env.display(cfg, model)
    elif wm.cfg.evaluate:
        while True:
            print("---------------------------")
            print(env.evaluate(cfg, model))
            print("---------------------------")


def setup_env(wm: Wingman):
    cfg = wm.cfg
    env = Environment(cfg)
    cfg.obs_size = env.obs_size
    cfg.act_size = env.act_size

    return env


def setup_nets(wm: Wingman):
    cfg = wm.cfg

    # set up networks and optimizers
    model = SAC(
        obs_size=cfg.obs_size,
        act_size=cfg.act_size,
        entropy_tuning=cfg.use_entropy,
        target_entropy=cfg.target_entropy,
        discount_factor=cfg.discount_factor,
    ).to(cfg.device)
    actor_optim = optim.AdamW(
        model.actor.parameters(), lr=cfg.learning_rate, amsgrad=True
    )
    critic_optim = optim.AdamW(
        model.critic.parameters(), lr=cfg.learning_rate, amsgrad=True
    )
    alpha_optim = optim.AdamW([model.log_alpha], lr=0.01, amsgrad=True)

    optims = dict()
    optims["actor"] = actor_optim
    optims["critic"] = critic_optim
    optims["alpha"] = alpha_optim

    # get latest weight files
    has_weights, model_file, optim_file = wm.get_weight_files()
    if has_weights:
        # load the model
        model.load_state_dict(
            torch.load(model_file, map_location=torch.device(cfg.device))
        )

        # load the optimizer
        checkpoint = torch.load(optim_file, map_location=torch.device(cfg.device))

        for opt_key in checkpoint:
            optims[opt_key].load_state_dict(checkpoint[opt_key])

    # torch.save(net.actor.net.state_dict(), f"./{set.env_name}_big.pth")
    # exit()

    return model, optims


if __name__ == "__main__":
    signal(SIGINT, shutdown_handler)
    wm = Wingman(config_yaml="./src/settings.yaml")

    """ SCRIPTS HERE """

    if wm.cfg.train:
        train(wm)
    elif wm.cfg.display or wm.cfg.evaluate:
        eval_display(wm)
    else:
        print("Guess this is life now.")
