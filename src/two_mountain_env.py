import time
import warnings

import gym
import numpy as np

from utils.helpers import cpuize, get_device, gpuize


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self):
        super().__init__()

        self.state = np.zeros((2,))
        self.state_size = self.state.shape[0]
        self.num_actions = 1
        self.do_nothing = np.zeros((1,))
        self.label = np.array([-1.0])

        self.total_steps = 1000
        self.steps = 0

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

        self.device = get_device()

        self.reset()

    def eval(self):
        self.eval_run = True

    def train(self):
        self.eval_run = False

    def reset(self):
        self.steps = 0
        self.done = 0
        self.cumulative_reward = 0

        self.state *= 0.0

    def get_state(self):
        label = self.get_label(self.state)
        return self.state, None, None, label

    def step(self, action):

        """
        actions are expected to be of shape [1]

        output:
            observations of shape [observation_shape]
            rewards of shape [1]
            dones of shape [1]
            labels of shape [num_actions]
        """
        action = action.squeeze()

        # the environment is just a line stretching from -100 to 100
        self.state[0] = np.clip(self.state[0] + action, -100.0, 100.0)
        self.state[1] = self.steps / self.total_steps

        # if we are distance 1 away from the left target, end the enviroment
        if abs(self.state[0] + 10.0) < 1.0:
            reward = -0.1
            self.done = True

        # if we are distance 1 away from the right target, end the environment
        elif abs(self.state[0] - 90.0) < 1.0:
            reward = 100.0
            self.done = True

        # otherwise just continue
        else:
            reward = -0.1
            self.done = False

        # accumulate rewards
        self.cumulative_reward += reward

        # get label
        label = self.get_label(self.state)

        # check for time truncation
        if self.steps > self.total_steps:
            self.done = True

        self.steps += 1

        return self.state, reward, self.done, label

    @property
    def is_done(self):
        """
        Enforce one way property of done
        """
        return self.done

    def get_label(self, obs):
        return self.label

    def evaluate(self, set, net=None):
        if net is not None:
            net.eval()
        self.eval()
        self.reset()

        eval_perf = []

        while len(eval_perf) < set.eval_num_episodes:
            # get the initial state and action
            obs, _, _, lbl = self.get_state()

            if net is not None:
                output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                action = net.actor.infer(*output)
                action = cpuize(action)[0]
            else:
                action = lbl

            # get the next state and reward
            _, _, _, _ = self.step(action)

            if self.is_done:
                eval_perf.append(self.cumulative_reward)
                self.reset()

        eval_perf = np.mean(np.array(eval_perf))
        return eval_perf

    def display(self, set, net=None):

        if net is not None:
            net.eval()
        self.eval()
        self.reset()

        action = self.do_nothing

        while True:
            obs, rwd, dne, lbl = self.step(action)

            if self.is_done:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset()
                action = self.do_nothing

            if net is not None:
                output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
            else:
                action = lbl

            # this is the state lol
            print(self.state)

            time.sleep(0.03)
