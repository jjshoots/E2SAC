import os
import cv2
import glob
import math
import numpy as np

from utils.helpers import cpuize, gpuize
from railway_drone.environment.aviary import Aviary


class Environment:
    """
    Wrapper for Aviary and Drone Classes with domain randomization
    """

    def __init__(self, image_size=(64, 64)):

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        tex_dir = current_file_path + "/../railway_drone/models/textures/"
        self.texture_paths = glob.glob(
            os.path.join(tex_dir, "**", "*.jpg"), recursive=True
        )

        self.max_steps = 1000
        self.image_size = image_size

        self.num_actions = 2
        self.state = np.zeros((1, *self.image_size))

        self.done = 0
        self.cumulative_reward = 0

        self.eval_run = False

    def eval(self):
        self.eval_run = True
        self.max_steps = math.inf

    def train(self):
        self.eval_run = False
        self.max_steps = 1000

    def reset(self, render=False):
        try:
            self.env.disconnect()
        except:
            pass

        self.done = 0
        self.step_count = 0
        self.cumulative_reward = 0

        self.env = Aviary(render=render, image_size=self.image_size)
        self.env.drone.set_mode(4)
        self.image_size = self.env.drone.frame_size
        self.state = np.zeros((4, *self.image_size))

        # clear argsv[0] message, I don't know why it works don't ask me why it works
        print("\033[A                             \033[A")

        self.update_textures()

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()
            self.env.drone.setpoint = np.array([0, 0, 0, 2])

        # track state is dist, angle
        # drone state is xy velocity
        self.track_state = self.env.track_state()
        self.drone_state = self.env.drone.state[-2][:2]

    def get_state(self):
        return self.drone_image, 0.0, 0.0, self.track_state

    def step(self, action):
        """
        step the entire simulation
            input is the railstate as [pos, orn]
            output is tuple of observation(ndarray), reward(scalar), done(int), trackstate([pos, orn])
        """
        reward = 0.0
        while not self.env.step():
            # reward is computed for the previous time step
            reward = -np.linalg.norm(self.track_state)
            self.cumulative_reward += reward

            # step the env
            self.env.drone.setpoint = self.tgt2set(action * self.env.track_state_norm)
            self.step_count += 1

            # every 240 time steps (1 seconds), change the texture of the floor
            if self.step_count % 240 == 1:
                self.update_textures()

            # get the states
            self.track_state = self.env.track_state()
            self.drone_state = self.env.drone.state[-2][:2]

            # check terminate
            self.done = 1.0 if self.step_count >= self.max_steps else 0.0
            if np.isnan(np.sum(self.track_state)):
                self.done = 1.0
                self.track_state = np.array([0.0, 0.0])

        # label = self.track_state if np.linalg.norm(self.track_state) > 0.4 else np.zeros_like(self.track_state)
        # label = np.clip(self.track_state, 0., 1.)
        label = self.track_state

        return self.drone_image, reward, self.done, label

    @property
    def is_done(self):
        """
        Enforce one way property of done
        """
        return self.done

    @property
    def drone_image(self):
        return self.transform_obs(self.env.drone.rgbImg)

    def transform_obs(self, obs):
        """
        resize and norm
        """
        obs = (obs - 127.5) / 127.5
        obs = np.transpose(obs, (2, 0, 1))

        return obs

    def tgt2set(self, track_state: np.ndarray) -> np.ndarray:
        gain = 2.0

        c = np.cos(track_state[1])
        s = np.sin(track_state[1])
        rot = np.array([[c, -s], [s, c]])

        setpoint = np.matmul(rot, np.array([[-gain * track_state[0]], [6.0]])).flatten()

        setpoint = np.array([*setpoint, gain * track_state[1], 2.0])

        return setpoint

    def update_textures(self):
        """
        randomly change the texture of the env
        25% chance of the rail being same texture as floor
        25% chance of clutter being same texture as rails
        25% chance of rail, floor, and clutter being same texture
        25% chance of all different
        """

        chance = np.random.randint(4)

        if chance == 0:
            # rail same as floor, clutter diff
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)

            tex_id = self.get_random_texture()
            self.env.clutter.change_rail_texture(tex_id)
        elif chance == 1:
            # rail same as clutter, floor diff
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)
            self.env.clutter.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)
        elif chance == 2:
            # all same
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)
            self.env.clutter.change_rail_texture(tex_id)
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)
        else:
            # all diff
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            self.env.clutter.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)

    def get_random_texture(self) -> int:
        texture_path = self.texture_paths[
            np.random.randint(0, len(self.texture_paths) - 1)
        ]
        tex_id = -1
        while tex_id < 0:
            tex_id = self.env.loadTexture(texture_path)
        return tex_id

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
        self.reset(render=True)

        action = np.zeros((set.num_actions))

        while True:
            obs, rwd, dne, lbl = self.step(action)

            if self.is_done:
                print(f"Total Reward: {self.cumulative_reward}")
                self.reset(render=True)
                action = np.zeros((set.num_actions))

            if net is not None:
                output = net.actor(gpuize(obs, set.device).unsqueeze(0))
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))[0]

                # print(action)
                print(
                    net.critic.forward(
                        gpuize(obs, set.device).unsqueeze(0),
                        net.actor.infer(*output)[0],
                    ).squeeze()
                )
            else:
                action = lbl
