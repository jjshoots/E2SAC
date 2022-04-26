import time
import cv2
import gym
import numpy as np

from utils.helpers import cpuize, gpuize


class Environment:
    """
    Wrapper for OpenAI gym environments that outputs suboptimal actions also
    """

    def __init__(self, image_size=(64, 64)):
        super().__init__()

        self.image_size = image_size
        self.frame_stack = 4

        self.env = gym.make("CarRacing-v1", verbose=False, continuous=False, perky=True)
        self.state = np.zeros((1, *self.image_size))
        self.num_actions = self.env.action_space.n

        self.off_track_t = 0
        self.max_off_track = 50

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

        self.reset()

    def switchup(self):
        self.env = gym.make(
            "CarRacing-v1", verbose=False, continuous=False, perky=True
        )

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
            self.step(0, startup=True)

        self.state = np.concatenate(
            [self.transform_obs(self.env.env.state)] * self.frame_stack, 0
        )

    def get_state(self):
        label = self.get_label(self.transform_obs(self.env.env.state))
        return self.state, None, None, label

    def step(self, action, startup=False):

        """
        actions are expected to be of shape [1]

        output:
            observations of shape [observation_shape]
            rewards of shape [1]
            dones of shape [1]
            labels of shape [num_actions]
        """
        action = int(np.squeeze(action))

        # step through the env for frame stack num times
        obs = []
        rwd = 0.0
        lbl = None
        for i in range(self.frame_stack):
            observation, reward, done, _ = self.env.step(action)
            obs.append(self.transform_obs(observation))
            rwd += reward
            self.done = max(done, self.done)

            if i == 0:
                lbl = (
                    None if startup else self.get_label(self.transform_obs(observation))
                )
        self.state = np.concatenate(obs, axis=0)

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
        return 0

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

    def display(self, set, net=None, transformed=False):

        if net is not None:
            net.eval()
        self.eval()
        self.reset()

        action = 0

        if transformed:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)

        while True:
            obs, rwd, dne, lbl = self.step(action)

            if self.is_done:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset()
                action = 0

            if net is not None:
                output = net.forward(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.sample(*output))
                action = cpuize(net.infer(*output))

                # print(action)
                # print(output.squeeze())
            else:
                action = lbl

            if transformed:
                display = obs[:3, ...]
                display = np.uint8((display * 127.5 + 127.5))
                display = np.transpose(display, (1, 2, 0))
                cv2.imshow("display", display)
                cv2.waitKey(int(1000 / 15))
            else:
                self.env.render()
                time.sleep(0.03)
