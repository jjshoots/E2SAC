import gymnasium as gym
import numpy as np
from PyFlyt.core.PID import PID
from wingman import cpuize, gpuize


class Environment:
    """
    Wrapper for gymnasium environments that outputs suboptimal actions also
    """

    def __init__(self, cfg):
        super().__init__()

        # make the env
        self.env = gym.make(
            cfg.env_name,
            render_mode=("human" if cfg.display else None),
            angle_representation="euler",
            use_yaw_targets=False,
            num_targets=10,
        )

        # compute spaces
        if len(self.env.action_space.shape) == 1:
            self.act_size = self.env.action_space.shape[0]
        else:
            raise NotImplementedError("Unsure how to deal with action space.")

        if len(self.env.observation_space.shape) == 1:
            self.obs_size = self.env.observation_space.shape[0]
        else:
            raise NotImplementedError("Unsure how to deal with observation space.")

        # constants
        self.device = cfg.device
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_range = (action_high - action_low)

        self.setup_oracle()

    def setup_oracle(self):
        # control period of the underlying controller
        self.ctrl_period = 1.0 / 120.0

        # input: angular position command
        # output: angular velocity
        Kp_ang_pos = np.array([3.0, 3.0])
        Ki_ang_pos = np.array([0.0, 0.0])
        Kd_ang_pos = np.array([0.0, 0.0])
        lim_ang_pos = np.array([1.0, 1.0])

        # input: linear velocity command
        # output: angular position
        Kp_lin_vel = np.array([1.0, 1.0])
        Ki_lin_vel = np.array([1.0, 1.0])
        Kd_lin_vel = np.array([1.0, 1.0])
        lim_lin_vel = np.array([0.4, 0.4])

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
        self.z_PID = PID(0.15, 1.0, 0.015, 0.3, self.ctrl_period)

    def compute_PIDs(self):
        ang_pos = self.state[3:6]
        lin_vel = self.state[6:9]
        setpoint = self.state[12:15]

        output = self.PIDs[1].step(lin_vel[:2], setpoint[:2])
        output = np.array([-output[1], output[0]])
        output = self.PIDs[0].step(ang_pos[:2], output)

        z_output = self.z_PID.step(lin_vel[-1], setpoint[-1])
        z_output = np.clip(z_output, 0.0, 1.0)

        self.pid_output = np.array([*output, 0.0, z_output])

        # normalize
        self.pid_output = (self.pid_output - self._action_mid) * self._action_range

    def get_label(self, obs):
        return self.pid_output

    def reset(self):
        self.state, _ = self.env.reset()

        self.ended = False
        self.cumulative_reward = 0

        self.setup_oracle()
        self.compute_PIDs()

        return self.state

    def step(self, action):
        action = action.squeeze()
        assert (
            action.shape[0] == self.act_size
        ), f"Incorrect action sizes, expected {self.act_size}, got {action.shape[0]}"

        # denormalize the action
        action = action / self._action_range + self._action_mid

        # step through the env
        self.state, reward, term, trunc, info = self.env.step(action)

        # accumulate rewards
        self.cumulative_reward += reward

        if term or trunc:
            self.ended = True

        # compute the pids so we don't have discontinuity
        self.compute_PIDs()

        return self.state, reward, term

    def evaluate(self, cfg, net=None):
        if net is not None:
            net.eval()

        # reset the env
        self.reset()

        # store the list of eval performances here
        eval_perf = []

        while len(eval_perf) < cfg.eval_num_episodes:

            # get the action based on the state
            if net is not None:
                output = net.actor(gpuize(self.state, cfg.device).unsqueeze(0))
                action = net.actor.infer(*output)
                action = cpuize(action)[0]
            else:
                action = self.get_label(self.state)

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

            if net is not None:
                output = net.actor(gpuize(self.state, cfg.device).unsqueeze(0))
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
            else:
                action = self.get_label(self.state)

            self.step(action)

            if self.ended:
                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print("-----------------------------------------")
                self.reset()
