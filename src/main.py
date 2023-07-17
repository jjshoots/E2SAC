import math
from signal import SIGINT, signal

import torch
import torch.optim as optim
from rail_env import RailEnv
from wingman import ReplayBuffer, Wingman, cpuize, gpuize, shutdown_handler

from algorithms import CCGE


def train(wm: Wingman):
    # grab config
    cfg = wm.cfg

    # setup env, model, replaybuffer
    env = setup_env(wm)
    model, optims = setup_nets(wm)
    memory = ReplayBuffer(cfg.buffer_size)

    wm.log["epoch"] = 0
    wm.log["eval_perf"] = -math.inf
    wm.log["max_eval_perf"] = -math.inf
    next_eval_step = 0

    while memory.count <= cfg.total_steps:
        wm.log["epoch"] += 1

        """EVALUATE POLICY"""
        if memory.count >= next_eval_step:
            next_eval_step = (
                int(memory.count / cfg.eval_steps_ratio) + 1
            ) * cfg.eval_steps_ratio
            wm.log["eval_perf"] = env.evaluate(cfg, model)
            wm.log["max_eval_perf"] = max(
                [float(wm.log["max_eval_perf"]), float(wm.log["eval_perf"])]
            )

        """ENVIRONMENT ROLLOUT"""
        env.reset()
        model.eval()
        model.zero_grad()

        with torch.no_grad():
            while not (env.ended):
                # get the initial state and label
                obs_att = env.obs_att
                obs_img = env.obs_img
                lbl = env.label

                if memory.count < cfg.exploration_steps:
                    act = env.env.action_space.sample()
                else:
                    # move observation to gpu
                    t_obs_att = gpuize(obs_att, cfg.device)
                    t_obs_img = gpuize(obs_img, cfg.device)

                    # get the action from policy
                    output = model.actor(t_obs_att, t_obs_img)
                    t_act, _ = model.actor.sample(*output)

                    # move label to gpu
                    t_lbl = gpuize(lbl, cfg.device)

                    # figure out whether to follow policy or oracle
                    sup_scale, *_ = model.calc_sup_scale(
                        t_obs_att, t_obs_img, t_act, t_lbl
                    )
                    act = lbl if sup_scale.squeeze(0) == 1.0 else cpuize(t_act)

                # get the next state and other stuff
                next_obs_att, next_obs_img, rew, term = env.step(act)

                # store stuff in mem
                memory.push(
                    [
                        obs_att,
                        obs_img,
                        act,
                        rew,
                        next_obs_att,
                        next_obs_img,
                        term,
                        lbl,
                    ],
                    random_rollover=cfg.random_rollover,
                )

            # for logging
            wm.log["total_reward"] = env.cumulative_reward

        """TRAINING RUN"""
        dataloader = torch.utils.data.DataLoader(
            memory, batch_size=cfg.batch_size, shuffle=True, drop_last=False
        )

        for repeat_num in range(int(cfg.repeats_per_buffer)):
            for batch_num, stuff in enumerate(dataloader):
                model.train()

                obs_att = gpuize(stuff[0], cfg.device)
                obs_img = gpuize(stuff[1], cfg.device)
                actions = gpuize(stuff[2], cfg.device)
                rewards = gpuize(stuff[3], cfg.device)
                next_obs_att = gpuize(stuff[4], cfg.device)
                next_obs_img = gpuize(stuff[5], cfg.device)
                terms = gpuize(stuff[6], cfg.device)
                labels = gpuize(stuff[7], cfg.device)

                # train critic
                for _ in range(cfg.critic_update_multiplier):
                    model.zero_grad()
                    q_loss, log = model.calc_critic_loss(
                        obs_att,
                        obs_img,
                        actions,
                        rewards,
                        next_obs_att,
                        next_obs_img,
                        terms,
                    )
                    wm.log = {**wm.log, **log}
                    q_loss.backward()
                    optims["critic"].step()
                    model.update_q_target()

                # train actor
                for _ in range(cfg.actor_update_multiplier):
                    model.zero_grad()
                    rnf_loss, log = model.calc_actor_loss(
                        obs_att, obs_img, terms, labels
                    )
                    wm.log = {**wm.log, **log}
                    rnf_loss.backward()
                    optims["actor"].step()

                    # train entropy regularizer
                    if model.use_entropy:
                        model.zero_grad()
                        ent_loss, log = model.calc_alpha_loss(obs_att, obs_img)
                        wm.log = {**wm.log, **log}
                        ent_loss.backward()
                        optims["alpha"].step()

                """WANDB"""
                wm.log["num_transitions"] = memory.count
                wm.log["buffer_size"] = memory.__len__()

                """WEIGHTS SAVING"""
                to_update, model_file, optim_file = wm.checkpoint(
                    loss=-float(wm.log["eval_perf"]), step=wm.log["num_transitions"]
                )
                if to_update:
                    torch.save(model.state_dict(), model_file)

                    optim_dict = dict()
                    for key in optims:
                        optim_dict[key] = optims[key].state_dict()
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
    env = RailEnv(cfg)
    cfg.obs_att_size = env.obs_att_size
    cfg.obs_img_size = env.obs_img_size
    cfg.act_size = env.act_size

    return env


def setup_nets(wm: Wingman):
    cfg = wm.cfg

    # set up networks and optimizers
    model = CCGE(
        act_size=cfg.act_size,
        obs_att_size=cfg.obs_att_size,
        obs_img_size=cfg.obs_img_size,
        entropy_tuning=cfg.use_entropy,
        target_entropy=cfg.target_entropy,
        discount_factor=cfg.discount_factor,
        confidence_lambda=cfg.confidence_lambda,
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

    # torch.save(model.actor.net.state_dict(), "./wing.pth")
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
