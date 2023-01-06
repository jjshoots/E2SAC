import math
from signal import SIGINT, signal

import torch
import torch.optim as optim
from wingman import ReplayBuffer, Wingman, cpuize, gpuize, shutdown_handler

from algorithms import CCGE
from d4rl_env import Environment


def train(wm: Wingman):
    # grab config
    cfg = wm.cfg

    # setup env, model, replaybuffer
    env = setup_env(wm)
    net, optim_set = setup_nets(wm)
    memory = ReplayBuffer(cfg.buffer_size)

    # setup the oracle
    if not cfg.pretrained_oracle:
        env.update_oracle_weights(net.actor.net.state_dict())
        print("Using from scratch oracle.")

    wm.log["epoch"] = 0
    wm.log["eval_perf"] = -math.inf
    wm.log["max_eval_perf"] = -math.inf
    wm.log["oracle_eval_perf"] = env.evaluate(cfg, None)
    next_eval_step = 0

    while memory.count <= cfg.total_steps:
        wm.log["epoch"] += 1

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio
            wm.log["eval_perf"] = env.evaluate(cfg, net)
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )

        """UPDATE ORACLE"""
        if cfg.update_oracle:
            if wm.log["eval_perf"] > wm.log["oracle_eval_perf"]:
                env.update_oracle_weights(net.actor.net.state_dict())
                wm.log["oracle_eval_perf"] = wm.log["eval_perf"]
                print("Found new best oracle, updating.")

        """ENVIRONMENT ROLLOUT"""
        env.reset()
        net.eval()
        net.zero_grad()

        with torch.no_grad():
            while not env.ended:
                # get the initial state and label
                obs = env.state
                lbl = env.get_label(obs)

                if memory.count < cfg.exploration_steps:
                    act = env.env.action_space.sample()
                else:
                    # move observation to gpu
                    t_obs = gpuize(obs, cfg.device)

                    # get the action from policy
                    output = net.actor(t_obs)
                    t_act, _ = net.actor.sample(*output)

                    # move label to gpu
                    t_lbl = gpuize(lbl, cfg.device)

                    # figure out whether to follow policy or oracle
                    sup_scale, *_ = net.calc_sup_scale(t_obs, t_act, t_lbl)
                    act = lbl if sup_scale.squeeze(0) == 1.0 else cpuize(t_act)

                # get the next state and other stuff
                next_obs, rew, term = env.step(act)

                # store stuff in mem
                memory.push([obs, act, rew, next_obs, term, lbl])

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
                labels = gpuize(stuff[5], cfg.device)

                # train critic
                for _ in range(cfg.critic_update_multiplier):
                    net.zero_grad()
                    q_loss, log = net.calc_critic_loss(
                        states, actions, rewards, next_states, terms
                    )
                    wm.log = {**wm.log, **log}
                    q_loss.backward()
                    optim_set["critic"].step()
                    net.update_q_target()

                # train actor
                for _ in range(cfg.actor_update_multiplier):
                    net.zero_grad()
                    rnf_loss, log = net.calc_actor_loss(states, terms, labels)
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
                    torch.save(opt_dict, optim_file)


def eval_display(wm: Wingman):
    cfg = wm.cfg
    env = setup_env(wm)

    if False:
        net, _ = setup_nets(wm)
    else:
        net = None

    if wm.cfg.display:
        env.display(cfg, net)
    elif wm.cfg.evaluate:
        while True:
            print(env.evaluate(cfg, net))


def setup_env(wm: Wingman):
    cfg = wm.cfg
    env = Environment(cfg)
    cfg.obs_size = env.obs_size
    cfg.act_size = env.act_size

    return env


def setup_nets(wm: Wingman):
    cfg = wm.cfg

    # set up networks and optimizers
    net = CCGE(
        act_size=cfg.act_size,
        obs_size=cfg.obs_size,
        entropy_tuning=cfg.use_entropy,
        target_entropy=cfg.target_entropy,
        discount_factor=cfg.discount_factor,
        confidence_lambda=cfg.confidence_lambda,
    ).to(cfg.device)
    actor_optim = optim.AdamW(
        net.actor.parameters(), lr=cfg.learning_rate, amsgrad=True
    )
    critic_optim = optim.AdamW(
        net.critic.parameters(), lr=cfg.learning_rate, amsgrad=True
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
        opt_dict = torch.load(optim_file)
        for opt_key in opt_dict:
            optim_set[opt_key].load_state_dict(opt_dict[opt_key])

    # torch.save(net.actor.net.state_dict(), f"./{set.env_name}_big.pth")
    # exit()

    return net, optim_set


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
