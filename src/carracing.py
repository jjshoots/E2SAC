import copy
import gym
import cv2
import numpy as np


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self, image_size=(64, 64), verbose=False):
        super().__init__()

        self.image_size = image_size
        self.frame_stack = 4
        self.max_off_track = 50

        self.env = gym.make("CarRacing-v0", verbose=verbose)
        self.num_actions = 2
        self.max_accel = 0.3
        self.max_brake = 0.5

        self.off_track_t = 0
        self.done_marker = 0
        self.cumulative_reward = 0
        self.state = np.zeros((1, *self.image_size))

        self.eval_run = False

        self.reset()

    def eval(self):
        self.eval_run = True

    def train(self):
        self.eval_run = False

    def reset(self):
        self.env.reset()
        for _ in range(50):
            self.off_track_t = 0
            self.step(np.zeros(self.num_actions), startup=True)

        self.off_track_t = 0
        self.done_marker = 0
        self.cumulative_reward = 0
        self.state = np.concatenate(
            [self.transform_obs(self.env.env.state)] * self.frame_stack, 0
        )

    def get_state(self):
        label = self.get_label(self.transform_obs(self.env.env.state))
        return self.state, None, None, label

    def step(self, action, startup=False):

        """
        actions are expected to be of shape [2]

        output:
            observations of shape [observation_shape]
            rewards of shape [1]
            dones of shape [1]
            labels of shape [num_actions]
        """

        steer = action[0]
        accel = ((action[1]) > 0) * abs(action[1]) * self.max_accel
        brake = ((action[1]) < 0) * abs(action[1]) * self.max_brake
        action = np.array([steer, accel, brake])

        obs, rwd, dne, _ = self.env.step(action)
        self.done_marker = copy.copy(dne)

        obs = self.transform_obs(obs)
        self.state = np.concatenate([self.state[1:], obs])

        label = None if startup else self.get_label(obs)

        if rwd < 0.0:
            self.off_track_t += 1
        else:
            self.off_track_t = 0

        # during training, only end when we go off track for more than specified steps
        if not self.eval_run:
            if self.off_track_t >= self.max_off_track:
                dne = 1.0
            else:
                dne = 0.0

        self.cumulative_reward += rwd

        return self.state, rwd, dne, label

    @property
    def is_done(self):
        """
        check for eval or training end criterias
        """
        eval_criteria = self.done_marker and self.eval_run
        train_criteria = self.off_track_t >= self.max_off_track and not self.eval_run
        return eval_criteria or train_criteria

    def transform_obs(self, obs):
        """
        grayscale and crop and resize and norm
        """
        obs = obs[:80, ...]
        obs = cv2.resize(obs, dsize=self.image_size, interpolation=cv2.INTER_LINEAR)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, 0)
        obs = (obs - 127.5) / 127.5
        obs = np.transpose(obs, (0, 2, 1))

        return obs

    def get_label(self, obs):
        obs = np.uint8((obs * 127.5 + 127.5))
        obs = np.transpose(obs, (2, 1, 0))

        # blur
        obs = cv2.GaussianBlur(obs, (5, 5), 0)
        # detect edges
        obs = cv2.Canny(obs, 50, 150)
        # cv2.imshow('something', obs)
        # cv2.waitKey(100000)
        # crop the image just above the car
        obs = obs[38:40, 15:49]
        # find the centre of the track
        obs = cv2.findNonZero(obs)

        if obs is not None:
            obs = (
                (obs[:, 0, 0].max() + obs[:, 0, 0].min())
                / 2
                / (self.image_size[1] - 30)
            )
            obs = (obs - 0.5) / 0.5
        else:
            obs = 0.0

        steering = 2.0 * obs
        accel = 0.5

        return np.clip(np.array([steering, accel]), -0.99, 0.99)
