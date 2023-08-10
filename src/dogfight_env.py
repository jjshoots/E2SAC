import cv2
import numpy as np
from PIL import Image
from pyflyt_dogfight import DogfightEnv
from wingman import cpuize, gpuize

from xbox_controller import XboxController


class Environment:
    """
    Wrapper for gymnasium environments that outputs suboptimal actions also
    """

    def __init__(self, cfg):
        super().__init__()

        self.env = DogfightEnv(
            agent_hz=60 if cfg.human else cfg.agent_hz,
            render=cfg.display,
            human_camera=cfg.human,
        )

        # compute spaces
        self.act_size = self.env.action_space.shape[0]
        self.obs_size = self.env.observation_space.shape[0]

        # constants
        self.device = cfg.device
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_range = (action_high - action_low) / 2.0

    def reset(self):
        self.obs, _ = self.env.reset()

        self.ended = False
        self.cumulative_reward = np.zeros((self.env.num_drones))

        return self.obs

    def step(self, action) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # denormalize the action
        action = action * self._action_range + self._action_mid

        # step through the env
        self.obs, rew, term, trunc, self.infos = self.env.step(action)

        # accumulate rewards
        self.cumulative_reward += rew

        if term.any() or trunc.any():
            self.ended = True

        return self.obs, rew, term

    def evaluate(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        # store the list of eval performances here
        eval_perf = []

        while len(eval_perf) < cfg.eval_num_episodes:
            obs = gpuize(self.obs, cfg.device)

            # get the action based on the state
            if net is not None:
                output = net.actor(obs)
                actions = net.actor.infer(*output)
                actions = cpuize(actions)
            else:
                actions = np.stack(
                    [
                        self.env.action_space.sample()
                        for _ in range(self.env.num_drones)
                    ],
                    axis=0,
                )

            self.step(actions)

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

        frames = []
        num_gifs = 0
        while True:
            obs = gpuize(self.obs, cfg.device)

            if net is not None:
                output = net.actor(obs)
                actions = cpuize(net.actor.infer(*output))
            else:
                actions = np.stack(
                    [
                        self.env.action_space.sample()
                        for _ in range(self.env.num_drones)
                    ],
                    axis=0,
                )

            self.step(actions)

            if cfg.make_gif:
                frames.append(self.env.render())

            if self.ended:
                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print(self.infos)
                print("-----------------------------------------")

                if cfg.make_gif:
                    frames = [Image.fromarray(f) for f in frames]
                    frames[0].save(
                        f"./gifs/gif{num_gifs}.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=(1000.0 / cfg.agent_hz),
                        loop=0,
                    )
                    frames = []
                    num_gifs += 1

                self.reset()

    def human(self, cfg, net):
        assert net is not None, "cannot fight against nothing!"
        controller = XboxController()

        # reset the env
        self.reset()

        frame_time = int(1000/60)
        while True:
            # get human and agent action
            obs = gpuize(self.obs[1], cfg.device).unsqueeze(0)
            output = net.actor(obs)
            agent_action = cpuize(net.actor.infer(*output))
            human_action = controller.read()
            actions = np.stack([human_action, agent_action[0]], axis=0)

            self.step(actions)

            # get the image from the environment
            image = self.env.render()

            # display the image using cv2
            cv2.imshow("fight!", image)
            cv2.waitKey(frame_time)

            if self.ended:
                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print(self.infos)
                print("-----------------------------------------")
                self.reset()
