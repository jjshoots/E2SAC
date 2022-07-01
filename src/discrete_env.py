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

        self.env = gym.make(
            "CarRacing-v1", verbose=False, continuous=False, domain_randomize=True
        )
        self.state = np.zeros((1, *self.image_size))
        self.num_actions = self.env.action_space.n

        self.do_nothing = 0

        self.off_track_t = 0
        self.max_off_track = 50

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

        self.reset()

    def eval(self):
        self.eval_run = True

    def train(self):
        self.eval_run = False

    def reset(self, randomize=False):
        self.off_track_t = 0
        self.done = 0
        self.cumulative_reward = 0

        self.env.reset(options={"randomize": randomize})
        for _ in range(50):
            self.off_track_t = 0
            self.env.step(self.do_nothing)

        self.state = np.concatenate(
            [self.transform_obs(self.env.env.state)] * self.frame_stack, 0
        )

    def get_state(self):
        return self.state, None, None

    def step(self, action):

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
        observation = []
        reward = 0.0
        for i in range(self.frame_stack):
            obs, rwd, dne, _ = self.env.step(action)
            observation.append(self.transform_obs(obs))
            reward += rwd
            self.done = max(dne, self.done)

        self.state = np.concatenate(observation, axis=0)

        # record the number of times we go off track or generate no rewards
        if reward < 0.0:
            self.off_track_t += 1
        else:
            self.off_track_t = 0

        # during training, only end when:
        #   - we go off track for more than specified steps (no reward)
        #   - or we go outside the map
        if not self.eval_run:
            if self.off_track_t >= self.max_off_track or reward < -50.0:
                self.done = 1.0
            else:
                self.done = 0.0

        # accumulate rewards
        self.cumulative_reward += reward

        return self.state, reward, self.done

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

    def evaluate(self, set, net=None):
        if net is not None:
            net.eval()
        self.eval()
        self.reset()

        eval_perf = []

        while len(eval_perf) < set.eval_num_traj:
            # get the initial state and action
            obs, _, _ = self.get_state()

            if net is not None:
                output = net(gpuize(obs, set.device).unsqueeze(0))
                action = cpuize(net.infer(*output))
            else:
                action = 0

            # get the next state and reward
            _, _, _ = self.step(action)

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

        action = self.do_nothing

        if transformed:
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)

        while True:
            obs, _, _ = self.step(action)

            if self.is_done:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset(randomize=True)
                action = self.do_nothing

            if net is not None:
                output = net.forward(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.sample(*output))
                action = cpuize(net.infer(*output))
            else:
                action = 0

            if transformed:
                display = obs[:3, ...]
                display = np.uint8((display * 127.5 + 127.5))
                display = np.transpose(display, (1, 2, 0))
                cv2.imshow("display", display)
                cv2.waitKey(int(1000 / 15))
            else:
                self.env.render("human")

    def evaluate_pre_post(self, set, net):
        net.eval()
        self.eval()

        # list of mean uncertainty for each episode
        episodic_uncertainty = []
        episodic_value = []
        # list of uncertainties for each step in an ep
        uncertainty = []
        value = []

        while len(episodic_uncertainty) < 50:
            # get the initial state and action
            obs, _, _ = self.get_state()

            q, f = net(gpuize(obs, set.device).unsqueeze(0))
            action = cpuize(net.infer(q, f))

            # get the next state and reward
            _, _, _ = self.step(action)

            # store the transition uncertainty
            uncertainty.append(cpuize(f.squeeze()))
            value.append(cpuize(q.squeeze()))

            if self.is_done:
                episodic_uncertainty.append(np.mean(np.asarray(uncertainty)))
                episodic_value.append(np.mean(np.asarray(value)))
                uncertainty = []
                value = []
                self.reset()

        return np.mean(np.asarray(episodic_uncertainty)), np.mean(
            np.asarray(episodic_value)
        )
