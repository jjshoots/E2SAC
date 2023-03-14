import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
import torch
from PIL import Image
from PyFlyt.core.abstractions import PID
from wingman import cpuize, gpuize

from suboptimal_policy import Suboptimal_Actor


class Environment:
    """
    Wrapper for gymnasium environments that outputs suboptimal actions also
    """

    def __init__(self, cfg):
        super().__init__()

        # environment params
        self.is_wing = "wing" in cfg.env_name
        self.num_targets = cfg.num_targets

        # make the env
        self.env = gym.make(
            cfg.env_name,
            render_mode=("human" if cfg.display else None),
            angle_representation="euler",
            num_targets=cfg.num_targets,
            agent_hz=cfg.agent_hz,
            sparse_reward=cfg.sparse_reward,
            render_resolution=(1920, 1080),
        )

        # compute spaces
        self.act_size = self.env.action_space.shape[0]
        self.context_length = cfg.context_length
        self.obs_atti_size = self.env.observation_space["attitude"].shape[0]
        self.obs_targ_size = self.env.observation_space[
            "target_deltas"
        ].feature_space.shape[0]

        # constants
        self.device = cfg.device
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_range = (action_high - action_low) / 2.0

        # suboptimal actor init
        self.suboptimal_actor = None

        self.setup_oracle()

    def setup_oracle(self):
        if not self.is_wing:
            # control period of the underlying controller
            self.ctrl_period = 1.0 / 30.0

            # grab the limits from the environment and downscale them
            a_lim = self.env.action_space.high[0] * 0.4
            t_lim = self.env.action_space.high[-1]

            # input: angular position command
            # output: angular velocity
            Kp_ang_pos = np.array([3.0, 3.0])
            Ki_ang_pos = np.array([0.0, 0.0])
            Kd_ang_pos = np.array([0.0, 0.0])
            lim_ang_pos = np.array([a_lim, a_lim])

            # input: linear velocity command
            # output: angular position
            Kp_lin_vel = np.array([1.0, 1.0])
            Ki_lin_vel = np.array([1.0, 1.0])
            Kd_lin_vel = np.array([1.0, 1.0])
            lim_lin_vel = np.array([1.0, 1.0])

            ang_pos_PID = PID(
                Kp_ang_pos,
                Ki_ang_pos,
                Kd_ang_pos,
                lim_ang_pos,
                self.ctrl_period,
            )
            lin_vel_PID = PID(
                Kp_lin_vel,
                Ki_lin_vel,
                Kd_lin_vel,
                lim_lin_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_pos_PID, lin_vel_PID]

            # height controllers
            self.z_PID = PID(0.3, 0.5, 0.0, t_lim, self.ctrl_period)
        else:
            # don't setup the oracle if it already exists
            if self.suboptimal_actor is not None:
                return

            suboptimal_path = f"./suboptimal_policies/wing.pth"
            weights = torch.load(suboptimal_path, map_location=self.device)

            # load the oracle
            self.suboptimal_actor = Suboptimal_Actor(
                obs_atti_size=self.obs_atti_size,
                obs_targ_size=self.obs_targ_size,
                context_length=self.context_length,
                act_size=self.act_size,
            ).to(self.device)
            self.suboptimal_actor.load_state_dict(weights)

            print(f"Loaded {suboptimal_path}")

    def compute_PIDs(self):
        if self.is_wing:
            return

        ang_pos = self.state_atti[3:6]
        lin_vel = self.state_atti[6:9]
        setpoint = self.state_targ[0]

        output = self.PIDs[1].step(lin_vel[:2], setpoint[:2])
        output = np.array([-output[1], output[0]])
        output = self.PIDs[0].step(ang_pos[:2], output)

        z_output = self.z_PID.step(lin_vel[-1], setpoint[-1])
        z_output = z_output.clip(min=0.0, max=self.z_PID.limits)

        self.pid_output = np.array([*output, 0.0, z_output])

        # normalize
        self.pid_output = (self.pid_output - self._action_mid) / self._action_range
        self.pid_output[-1] = np.clip(self.pid_output[-1], -0.3, 1.0)

    def get_label(self, state) -> np.ndarray:
        if not self.is_wing:
            return self.pid_output
        else:
            action = self.suboptimal_actor(*[gpuize(s, self.device) for s in state])
            action = cpuize(torch.tanh(action))[0]

            return action

    def reset(self):
        obs, _ = self.env.reset()

        # splice out the observation and mask the target deltas
        self.state_atti = obs["attitude"]
        self.state_targ = np.zeros((self.num_targets, 3))
        self.state_targ[: len(obs["target_deltas"])] = obs["target_deltas"]

        self.ended = False
        self.cumulative_reward = 0

        self.setup_oracle()
        self.compute_PIDs()

        return self.state_atti, self.state_targ

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
        self.state_atti = obs["attitude"]
        self.state_targ = np.zeros((self.num_targets, 3))
        self.state_targ[: len(obs["target_deltas"])] = obs["target_deltas"]

        # accumulate rewards
        self.cumulative_reward += rew

        if term or trunc:
            self.ended = True

        # compute the pids so we don't have discontinuity
        self.compute_PIDs()

        return self.state_atti, self.state_targ, rew, term

    def evaluate(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        # store the list of eval performances here
        eval_perf = []
        num_hits = []

        while len(eval_perf) < cfg.eval_num_episodes:
            state_atti = gpuize(self.state_atti, cfg.device).unsqueeze(0)
            state_targ = gpuize(self.state_targ, cfg.device).unsqueeze(0)

            # get the action based on the state
            if net is not None:
                output = net.actor(state_atti, state_targ)
                action = net.actor.infer(*output)
                action = cpuize(action)[0]
            else:
                action = self.get_label((state_atti, state_targ))

            self.step(action)

            if self.ended:
                eval_perf.append(self.cumulative_reward)
                num_hits.append(float(self.env.info["num_targets_reached"]))
                self.reset()

        eval_perf = np.mean(np.array(eval_perf))
        num_hits = np.mean(np.array(num_hits))
        return num_hits

    def display(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        gifs_save_path = "./rendered_gifs"
        total_gifs = 0
        num_steps = 0
        frames = []
        overlay = None

        while True:
            state_atti = gpuize(self.state_atti, cfg.device).unsqueeze(0)
            state_targ = gpuize(self.state_targ, cfg.device).unsqueeze(0)

            if net is not None:
                output = net.actor(state_atti, state_targ)
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
            else:
                action = self.get_label((state_atti, state_targ))

            # print(action, self.pid_output)
            self.step(action)
            num_steps += 1

            # this captures the camera image for gif
            if cfg.render_gif:
                frames.append(self.env.render()[..., :3].astype(np.uint8))

            # this captures the camera image for overlay trajectory
            if cfg.render_overlay and num_steps % 5 == 0:
                if overlay is None:
                    overlay = self.env.render()[..., :3]
                else:
                    overlay = np.min(np.stack([overlay, self.env.render()[..., :3]], axis=0), axis=0)

            if self.ended:
                if cfg.render_overlay:
                    from PIL import Image

                    num_steps = 0
                    im = Image.fromarray(overlay)
                    im.save("./quadx_trajectory.png")
                    exit()

                if cfg.render_gif:
                    print("-----------------------------------------")
                    print(f"Saving gif...")
                    print("-----------------------------------------")
                    frames = [Image.fromarray(frame) for frame in frames]
                    frames[0].save(
                        f"{gifs_save_path}/gif{total_gifs}.gif",
                        save_all=True,
                        append_images=frames[1:],
                        duration=1000 / 30,
                        loop=0,
                    )
                    frames = []
                    total_gifs += 1

                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print("-----------------------------------------")
                self.reset()
