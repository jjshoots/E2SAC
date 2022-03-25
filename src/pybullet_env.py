import time

import torch
import gym
import numpy as np
import pybulletgym

from utils.helpers import cpuize, gpuize, get_device
from suboptimal_policy import Suboptimal_Actor


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self, env_name):
        super().__init__()

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state = np.zeros_like(self.env.reset())
        self.state_size = self.state.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.do_nothing = np.zeros(self.num_actions)

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

        # load suboptimal policy
        self.device = get_device()
        try:
            self.suboptimal_actor = Suboptimal_Actor(
                num_actions=self.num_actions, state_size=self.state_size
            ).to(self.device)
            self.suboptimal_actor.load_state_dict(
                torch.load(f"./suboptimal_policies/{env_name}.pth")
            )
        except:
            print("--------------------------------------------------")
            print(f"No subotpimal actor found for env {env_name}")
            print("--------------------------------------------------")
            self.suboptimal_actor = None

        self.reset()

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
        actions are expected to be of shape [8]

        output:
            observations of shape [observation_shape]
            rewards of shape [1]
            dones of shape [1]
            labels of shape [num_actions]
        """
        action = action.squeeze()

        # step through the env
        self.state, reward, self.done, _ = self.env.step(action)

        # accumulate rewards
        self.cumulative_reward += reward

        # get label
        label = self.get_label(self.state)

        return self.state, reward, self.done, label

    @property
    def is_done(self):
        """
        Enforce one way property of done
        """
        return self.done

    def get_label(self, obs):
        if self.suboptimal_actor is not None:
            action = self.suboptimal_actor(gpuize(obs, self.device))
            action = cpuize(action)
            return action
        else:
            return self.do_nothing

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
        self.env = gym.make(self.env_name)
        self.env.render()
        self.eval()
        self.reset()

        action = np.zeros((set.num_actions))

        while True:
            obs, rwd, dne, lbl = self.step(action)

            if self.is_done:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset()
                action = np.zeros((set.num_actions))

            if net is not None:
                output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))

                # print(action)
                print(
                    net.critic.forward(
                        gpuize(obs, set.device).unsqueeze(0),
                        net.actor.infer(*output),
                    ).squeeze()
                )
            else:
                action = lbl

            time.sleep(0.03)
