import gym
import pybulletgym

import cv2
import numpy as np


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self, display=False):
        super().__init__()

        self.env = gym.make("AntPyBulletEnv-v0")
        self.min_actions = self.env.action_space.low
        self.max_actions = self.env.action_space.high
        self.num_actions = self.env.action_space.shape[0]

        self.a_mean = (self.max_actions + self.min_actions) / 2.0
        self.a_range = self.max_actions - self.min_actions

        if display:
            self.env.render()
        self.reset()

        self.state_size = 28

    def reset(self):
        self.env.reset()

    def get_state(self):
        self.state = self.env.robot.calc_state()
        return self.state, None, None, self.get_label(self.state)

    def step(self, action):

        """
        actions are expected to be of shape [num_envs, self.action_space.shape]

        output:
            observations of shape [observation_shape]
            rewards of shape [1]
            dones of shape [1]
            labels of shape [num_actions]
        """

        # rescale and shift
        action = action / (self.a_range / 2.0) + self.a_mean

        obs, rwd, dne, _ = self.env.step(action)
        self.state = obs

        return self.state, rwd, dne, self.get_label(obs)

    def get_label(self, obs):
        label = np.array([0, 1, 0, -1, 0, -1, 0, 1])

        # normalize and clip
        label = (label - self.a_mean) * (self.a_range / 2.0)
        label = np.clip(label, -0.99, 0.99)
        return label
