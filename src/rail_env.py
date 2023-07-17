import numpy as np
from pyflyt_rail_env import Environment
from wingman import cpuize, gpuize


class RailEnv:
    """
    Wrapper for gymnasium environments that outputs suboptimal actions also
    """

    def __init__(self, cfg):
        super().__init__()

        self.env = Environment(render_mode="human" if cfg.display else None)

        # compute spaces
        self.act_size = self.env.action_space.shape[0]
        self.obs_att_size = self.env.observation_space["attitude"].shape[0]
        self.obs_img_size = self.env.observation_space["rgba_img"].shape

        # constants
        self.device = cfg.device
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_range = (action_high - action_low) / 2.0

    def reset(self):
        obs, _ = self.env.reset()

        # splice out the observation and mask the target deltas
        self.obs_att = obs["attitude"]
        self.obs_img = obs["rgba_img"].transpose((2, 0, 1))

        self.ended = False
        self.cumulative_reward = 0

        return self.obs_att, self.obs_img

    @property
    def label(self) -> np.ndarray:
        label = np.zeros((self.act_size, ))
        track_position = self.env.track_state
        label[0] = 0.75
        label[1] = track_position[0]
        label[2] = track_position[1]
        label[3] = 1.0 - self.env.drone.state[-1][-1]
        return label

    def step(self, action) -> tuple[np.ndarray, np.ndarray, float, bool]:
        action = action.squeeze()
        assert (
            action.shape[0] == self.act_size
        ), f"Incorrect action sizes, expected {self.act_size}, got {action.shape[0]}"

        # denormalize the action
        action = action * self._action_range + self._action_mid

        # step through the env multiple times
        obs, rew, term, trunc, info = self.env.step(action)

        # splice out the observation and mask the target deltas
        self.obs_att = obs["attitude"]
        self.obs_img = obs["rgba_img"].transpose((2, 0, 1))

        # accumulate rewards
        self.cumulative_reward += rew

        if term or trunc:
            self.ended = True

        return self.obs_att, self.obs_img, rew, term

    def evaluate(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        # store the list of eval performances here
        eval_perf = []

        while len(eval_perf) < cfg.eval_num_episodes:
            obs_att = gpuize(self.obs_att, cfg.device).unsqueeze(0)
            obs_img = gpuize(self.obs_img, cfg.device).unsqueeze(0)

            # get the action based on the state
            if net is not None:
                output = net.actor(obs_att, obs_img)
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
            obs_att = gpuize(self.obs_att, cfg.device).unsqueeze(0)
            obs_img = gpuize(self.obs_img, cfg.device).unsqueeze(0)

            if net is not None:
                output = net.actor(obs_att, obs_img)
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
            else:
                action = self.label

            self.step(action)

            if self.ended:
                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print("-----------------------------------------")
                self.reset()
