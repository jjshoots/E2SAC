import warnings

import gymnasium as gym
import numpy as np
import torch
from wingman import cpuize, gpuize

from suboptimal_policy import Suboptimal_Actor

from gymnasium.spaces.utils import flatten
from gymnasium.spaces import Dict

class Environment:
    """
    Wrapper for D4RL Environments with suboptimal actions
    """

    def __init__(self, cfg):
        super().__init__()

        env_name = cfg.env_name.split("-")
        self.env_name = env_name[0] + ("Sparse-" if cfg.sparse else "-") + env_name[1]

        # make the env
        self.env = gym.make(
            self.env_name, render_mode=("human" if cfg.display else "rgb_array")
        )

        # flatten the space if needed
        if isinstance(self.env.observation_space, Dict):
            self.env = gym.wrappers.FlattenObservation(self.env)

        # compute spaces
        if len(self.env.action_space.shape) == 1:
            self.act_size = self.env.action_space.shape[0]
        else:
            raise NotImplementedError("Unsure how to deal with action space.")

        if len(self.env.observation_space.shape) == 1:
            self.obs_size = self.env.observation_space.shape[0]
        else:
            raise NotImplementedError("Unsure how to deal with observation space.")

        # constants
        self.device = cfg.device
        self.do_nothing = np.zeros((self.act_size))
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_scale = (action_high - action_low) / 2.0

        # load suboptimal policy
        suboptimal_path = f"./suboptimal_policies/{cfg.env_name}_{cfg.target_performance}.pth"
        try:
            # hacky way to get number of neurons per layer because I fked up with oracle training
            weights = torch.load(suboptimal_path)
            neurons_per_layer = weights["net.0.0.weight"].shape[0]

            # load the oracle
            self.suboptimal_actor = Suboptimal_Actor(
                act_size=self.act_size,
                obs_size=self.obs_size,
                neurons_per_layer=neurons_per_layer,
            ).to(cfg.device)
            self.suboptimal_actor.load_state_dict(weights)

            print(f"Loaded {suboptimal_path}")
        except FileNotFoundError:
            warnings.warn("--------------------------------------------------")
            warnings.warn(f"Failed to load suboptimal actor {suboptimal_path}, exiting.")
            warnings.warn("--------------------------------------------------")
            self.suboptimal_actor = None

        self.reset()

    def reset(self):
        self.state, _ = self.env.reset()

        self.ended = False
        self.cumulative_reward = 0

        return self.state

    def step(self, action) -> tuple[np.ndarray, float, bool]:
        action = action.squeeze()
        assert (
            action.shape[0] == self.act_size
        ), f"Incorrect action sizes, expected {self.act_size}, got {action.shape[0]}"

        # denormalize the action
        action = action * self._action_scale + self._action_mid

        # step through the env
        self.state, reward, term, trunc, info = self.env.step(action)

        # accumulate rewards
        self.cumulative_reward += reward

        if term or trunc:
            self.ended = True

        return self.state, reward, term

    def update_oracle_weights(self, weights: dict):
        assert self.suboptimal_actor is not None, "Can't update None model."
        self.suboptimal_actor.load_state_dict(weights)

    def get_label(self, obs):
        if self.suboptimal_actor is not None:
            action = self.suboptimal_actor(gpuize(obs, self.device))
            action = torch.tanh(action[0])
            action = cpuize(action)[0]
            return action
        else:
            return self.do_nothing

    def evaluate(self, cfg, net=None):
        if net is not None:
            net.eval()

        # make the env
        self.env.close()
        self.env = gym.make(self.env_name, render_mode="rgb_array")
        self.reset()

        # store the list of eval performances here
        eval_perf = []

        while len(eval_perf) < cfg.eval_num_episodes:

            # get the action based on the state
            if net is not None:
                output = net.actor(gpuize(self.state, cfg.device).unsqueeze(0))
                action = net.actor.infer(*output)
                action = cpuize(action)[0]
            else:
                action = self.get_label(self.state)

            # get the next state and reward
            self.step(action)

            if self.ended:
                eval_perf.append(self.cumulative_reward)
                self.reset()

        eval_perf = np.mean(np.array(eval_perf))
        return eval_perf

    def display(self, cfg, net=None):

        if net is not None:
            net.eval()

        # make the env
        self.env.close()
        self.env = gym.make(self.env_name, render_mode="human")
        self.reset()

        while True:

            if net is not None:
                output = net.actor(gpuize(self.state, cfg.device).unsqueeze(0))
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
                self.step(action)
            else:
                self.step(self.get_label(self.state))

            if self.ended:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset()
