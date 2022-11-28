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

        # environment params
        self.num_targets = cfg.num_targets

        # make the env
        self.env = gym.make(
            cfg.env_name,
            render_mode=("human" if cfg.display else None),
            angle_representation="euler",
            use_yaw_targets=False,
            num_targets=cfg.num_targets,
        )

        # compute spaces
        self.act_size = self.env.action_space.shape[0]
        self.obs_atti_size = self.env.observation_space["attitude"].shape[0]
        self.obs_targ_size = self.env.observation_space[
            "target_deltas"
        ].node_space.shape[0]

        # constants
        self.device = cfg.device
        action_high = self.env.action_space.high
        action_low = self.env.action_space.low
        self._action_mid = (action_high + action_low) / 2.0
        self._action_range = (action_high - action_low) / 2.0

        self.setup_oracle()

    def setup_oracle(self):
        # control period of the underlying controller
        self.ctrl_period = 1.0 / 30.0

        a_lim = self.env.action_space.high[0]
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
        self.z_PID = PID(0.15, 0.5, 0.0, t_lim, self.ctrl_period)

    def compute_PIDs(self):
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

    def get_label(self, *_):
        return self.pid_output

    def reset(self):
        obs, _ = self.env.reset()

        # splice out the observation and mask the target deltas
        self.state_atti = obs["attitude"]
        self.state_targ = np.zeros((self.num_targets, 3))
        self.state_targ[: len(obs["target_deltas"].nodes)] = obs["target_deltas"].nodes

        self.ended = False
        self.cumulative_reward = 0

        self.setup_oracle()
        self.compute_PIDs()

        return self.state_atti, self.state_targ

    def step(self, action):
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
        self.state_targ[: len(obs["target_deltas"].nodes)] = obs["target_deltas"].nodes

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

        while len(eval_perf) < cfg.eval_num_episodes:

            state_atti = gpuize(self.state_atti, cfg.device).unsqueeze(0)
            state_targ = gpuize(self.state_targ, cfg.device).unsqueeze(0)

            # get the action based on the state
            if net is not None:
                output = net.actor(state_atti, state_targ)
                action = net.actor.infer(*output)
                action = cpuize(action)[0]
            else:
                action = self.get_label()

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

            state_atti = gpuize(self.state_atti, cfg.device).unsqueeze(0)
            state_targ = gpuize(self.state_targ, cfg.device).unsqueeze(0)

            if net is not None:
                output = net.actor(state_atti, state_targ)
                # action = cpuize(net.actor.sample(*output)[0][0])
                action = cpuize(net.actor.infer(*output))
            else:
                action = self.get_label()

            # print(action, self.pid_output)

            self.step(action)

            if self.ended:
                print("-----------------------------------------")
                print(f"Total Reward: {self.cumulative_reward}")
                print("-----------------------------------------")
                self.reset()
