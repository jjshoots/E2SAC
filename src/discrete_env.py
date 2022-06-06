import time

import gym
import numpy as np

from utils.helpers import cpuize, gpuize


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self, env_name):
        super().__init__()

        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.state = np.zeros_like(self.env.reset())
        self.state_size = self.state.shape[0]
        self.num_actions = self.env.action_space.n
        self.do_nothing = 1

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

        self.has_switched = False

        self.reset()

    # def switchup(self):
    #     if not self.has_switched:
    #         self.env = gym.make(self.env_name, enable_wind=True, wind_power=10.0)
    #         self.has_switched = True

    def eval(self):
        self.eval_run = True

    def train(self):
        self.eval_run = False

    def reset(self):
        self.done = 0
        self.cumulative_reward = 0

        self.state = self.env.reset()

    def get_state(self):
        label = self.get_label(self.state)
        return self.state, None, None, label

    def step(self, action, startup=False):

        """
        actions are expected to be of shape 1

        output:
            observations of shape [observation_shape]
            rewards of shape [1]
            dones of shape [1]
            labels of shape [num_actions]
        """
        # remove array from action
        action = np.squeeze(action)

        # step through the env
        self.state, reward, self.done, info = self.env.step(action)

        # accumulate rewards
        self.cumulative_reward += reward

        # get label
        label = self.get_label(self.state)

        # only return done when we fail the env, not time limit
        done = self.done
        if "TimeLimit.truncated" in info:
            done = False

        return self.state, reward, done, label

    @property
    def is_done(self):
        """
        Enforce one way property of done
        """
        return self.done

    def get_label(self, obs):
        return self.do_nothing

    def evaluate(self, set, net=None):
        if net is not None:
            net.eval()
        self.eval()
        self.reset()

        eval_perf = []

        while len(eval_perf) < set.eval_num_traj:
            # get the initial state and action
            obs, _, _, lbl = self.get_state()

            if net is not None:
                output = net(gpuize(obs, set.device).unsqueeze(0))
                action = cpuize(net.infer(*output))
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
                output = net.forward(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.sample(*output))
                action = cpuize(net.infer(*output))

                # print(action)
                # print(output.squeeze())
            else:
                action = lbl

            self.env.render()
            time.sleep(0.03)