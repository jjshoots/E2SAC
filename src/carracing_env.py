import warnings

import cv2
import gym
import numpy as np
import torch

from suboptimal import SuboptimalActor
from utils.helpers import cpuize, gpuize, get_device


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self, image_size=(64, 64), randomize=True):
        super().__init__()

        self.image_size = image_size
        self.frame_stack = 4

        self.env = gym.make("CarRacing-v1", verbose=False, domain_randomize=randomize)
        self.state = np.zeros((1, *self.image_size))
        self.num_actions = 2

        self.off_track_t = 0
        self.max_off_track = 50

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

        self.device = get_device()

        self.use_learned_suboptimal = True

        # load suboptimal policy
        if self.use_learned_suboptimal:
            print("Using learned suboptimal policy")
            load_success = False
            path = f"./suboptimal_policies/suboptimal.pth"
            for _ in range(5):
                try:
                    self.suboptimal_actor = SuboptimalActor(
                        num_actions=self.num_actions
                    ).to(self.device)
                    self.suboptimal_actor.load_state_dict(torch.load(path))
                    print(f"Loaded {path}")
                    load_success = True
                except:
                    pass

                if load_success:
                    break

            if not load_success:
                warnings.warn("--------------------------------------------------")
                warnings.warn(f"Failed to load suboptimal actor {path}, exiting.")
                warnings.warn("--------------------------------------------------")
                exit()

        self.reset()

    def eval(self):
        self.eval_run = True

    def train(self):
        self.eval_run = False

    def reset(self):
        self.off_track_t = 0
        self.done = 0
        self.cumulative_reward = 0

        self.env.reset()
        for _ in range(50):
            self.off_track_t = 0
            self.step(np.zeros(self.num_actions), startup=True)

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
        # formulate the action
        steer = action[0]
        accel = ((action[1]) > 0) * abs(action[1])
        brake = ((action[1]) < 0) * abs(action[1])
        action = np.array([steer, accel, brake])

        # step through the env for frame stack num times
        obs = []
        rwd = 0.0
        lbl = None
        for i in range(self.frame_stack):
            observation, reward, done, _ = self.env.step(action)
            obs.append(self.transform_obs(observation))
            rwd += reward
            self.done = max(done, self.done)

        self.state = np.concatenate(obs, axis=0)
        lbl = None if startup else self.get_label(self.transform_obs(self.state[:3]))

        # record the number of times we go off track or generate no rewards
        if rwd < 0.0:
            self.off_track_t += 1
        else:
            self.off_track_t = 0

        # during training, only end when:
        #   - we go off track for more than specified steps (no reward)
        #   - or we go outside the map
        if not self.eval_run:
            if self.off_track_t >= self.max_off_track or rwd < -50.0:
                self.done = 1.0
            else:
                self.done = 0.0

        # accumulate rewards
        self.cumulative_reward += rwd

        return self.state, rwd, self.done, lbl

    @property
    def is_done(self):
        """
        Enforce one way property of done
        """
        return self.done

    def transform_obs(self, obs):
        """
        resize and norm
        """
        obs = cv2.resize(obs, dsize=self.image_size, interpolation=cv2.INTER_LINEAR)
        obs = (obs - 127.5) / 127.5

        obs = np.transpose(obs, (2, 0, 1))

        return obs

    def get_label(self, obs):
        if self.use_learned_suboptimal:
            action = self.suboptimal_actor(
                gpuize(self.state, self.device).unsqueeze(0)
            ).squeeze(0)
            action = torch.tanh(action[0])
            action = cpuize(action)[0]
            return action
        else:
            obs = obs[:, 15, :]
            obs = np.sum(obs, axis=0)
            obs = obs[1:] - obs[:-1]

            rise = np.argmax(obs)
            fall = np.argmin(obs)

            midpoint = (rise + fall) / 2.0
            midpoint = (midpoint / 64.0) - 0.5

            steering = midpoint
            accel = 0.1

            return np.clip(np.array([steering, accel]), -0.99, 0.99)

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

    def display(self, set, net=None, transformed=False):

        if net is not None:
            net.eval()
        self.eval()
        self.reset()

        action = np.zeros((set.num_actions))

        if transformed:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)

        while True:
            obs, rwd, dne, lbl = self.step(action)

            if self.is_done:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset()
                action = np.zeros((set.num_actions))

            if net is not None:
                output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))[0]

                print(
                    net.critic.forward(
                        gpuize(obs, set.device).unsqueeze(0),
                        net.actor.infer(*output)[0],
                    ).squeeze()
                )
            else:
                action = lbl

            if transformed:
                display = obs[:3, ...]
                display = np.uint8((display * 127.5 + 127.5))
                display = np.transpose(display, (1, 2, 0))
                cv2.imshow("display", display)
            else:
                self.env.render()

            cv2.waitKey(int(1000 / 15))
