import time

import matplotlib.pyplot as plt
import numpy as np
from pyflyt_rail_env import Environment
from wingman import cpuize, gpuize


class RailEnv:
    """
    Wrapper for gymnasium environments that outputs suboptimal actions also
    """

    def __init__(self, cfg):
        super().__init__()

        self.env = Environment(
            agent_hz=cfg.agent_hz, render_mode="human" if cfg.display else None
        )

        # compute spaces
        self.act_size = self.env.action_space.shape[0]
        self.obs_size = (
            self.env.observation_space["seg_img"].shape[2],
            self.env.observation_space["seg_img"].shape[0],
            self.env.observation_space["seg_img"].shape[1],
        )

        # constants
        self.device = cfg.device
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_range = (action_high - action_low) / 2.0

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()

        # splice out the observation and mask the target deltas
        self.obs = obs["seg_img"].transpose((2, 0, 1))

        self.infos = dict()
        self.ended = False
        self.cumulative_reward = 0
        self.reward_breakdown = 0.0

        return self.obs

    @property
    def label(self) -> np.ndarray:
        label = np.zeros((self.act_size,))
        track_position = self.env.track_state
        label[0] = 1.0
        label[1] = track_position[1]
        return label

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        action = action.squeeze()
        assert (
            action.shape[0] == self.act_size
        ), f"Incorrect action sizes, expected {self.act_size}, got {action.shape[0]}"

        # denormalize the action
        action = action * self._action_range + self._action_mid

        # step through the env multiple times
        obs, rew, term, trunc, self.infos = self.env.step(action)

        # splice out the observation and mask the target deltas
        self.obs = obs["seg_img"].transpose((2, 0, 1))

        # accumulate rewards
        self.cumulative_reward += rew

        if term or trunc:
            self.ended = True

        return self.obs, rew, term

    def evaluate(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        # store the list of eval performances here
        eval_perf = []

        while len(eval_perf) < cfg.eval_num_episodes:
            obs = gpuize(self.obs, cfg.device).unsqueeze(0)

            # get the action based on the state
            if net is not None:
                output = net.actor(obs)
                action = net.actor.infer(*output)
                action = cpuize(action)[0]
            else:
                action = self.label

            self.step(action)

            if self.ended:
                eval_perf.append(self.cumulative_reward)
                self.reset()

        eval_perf = np.mean(np.array(eval_perf))
        return eval_perf

    def display(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        while True:
            obs = gpuize(self.obs, cfg.device).unsqueeze(0)

            if net is not None:
                output = net.actor(obs)
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
            else:
                action = self.label

            self.step(action)

            if self.ended:
                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print(f"Infos: {self.infos}")
                print("-----------------------------------------")
                time.sleep(4)
                self.reset()
