import os
import cv2
import math
import numpy as np

from railway_drone.environment.PID import PID

import pybullet as p
from pybullet_utils import bullet_client


class Drone:
    def __init__(
        self,
        p: bullet_client.BulletClient,
        camera_FOV=90,
        frame_size=(128, 128),
        seg_ratio=4.0,
        ctrl_hz=48.0,
        sim_hz=240.0,
        cam_hz=24.0,
    ):
        # default physics looprate is 240 Hz
        self.p = p
        self.ctrl_period = 1.0 / ctrl_hz
        self.sim_period = 1.0 / sim_hz
        self.ctrl_update_ratio = int(sim_hz / ctrl_hz)
        self.cam_update_ratio = int(sim_hz / cam_hz)

        """ SPAWN """
        # spawn drone
        start_x = (np.random.rand() - 0.5) * 4
        start_x += np.sign(start_x) * 2
        start_y = (np.random.rand() - 0.5) * 8
        start_rot = np.random.rand() * np.sign(start_x) * 0.5
        # start_x = 0
        # start_y = 0
        # start_rot = 0

        self.start_pos = [start_x, start_y, 2]
        self.start_orn = self.p.getQuaternionFromEuler([0, 0, start_rot])

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        drone_dir = current_file_path + "/../models/vehicles/primitive_drone/drone.urdf"

        self.Id = self.p.loadURDF(
            drone_dir,
            basePosition=self.start_pos,
            baseOrientation=self.start_orn,
            useFixedBase=False,
        )

        """
        DRONE CONTROL
            motor_id corresponds to QuadrotorX in PX4
            control commands are in the form of pitch-roll-yaw-thrust
                using ENU convention
        """
        # the joint IDs corresponding to motorID 1234
        self.motor_id = np.array([0, 3, 2, 1])
        self.thr_coeff = np.array([[0.0, 0.0, 0.3]]) * 1e-4
        self.tor_coeff = np.array([[0.0, 0.0, 1.0]]) * 1e-4
        self.tor_dir = np.array([[1.0], [1.0], [-1.0], [-1.0]])
        self.noise_ratio = 0.02

        # maximum motor RPM
        self.max_rpm = 1700.0 * 25.2
        # motor modelled with first order ode, below is time const
        self.motor_tau = 0.01
        # motor mapping from angular torque to individual motors
        self.motor_map = np.array(
            [
                [+1.0, -1.0, +1.0, +1.0],
                [-1.0, +1.0, +1.0, +1.0],
                [+1.0, +1.0, -1.0, +1.0],
                [-1.0, -1.0, -1.0, +1.0],
            ]
        )

        # outputs normalized body torque commands
        self.Kp_ang_vel = np.array([0.5, 0.5, 0.05])
        self.Ki_ang_vel = np.array([0.005, 0.005, 0.0])
        self.Kd_ang_vel = np.array([0.001, 0.001, 0.001])
        self.lim_ang_vel = np.array([1.0, 1.0, 1.0])

        # outputs angular rate
        self.Kp_ang_pos = np.array([10.0, 10.0, 50.0])
        self.Ki_ang_pos = np.array([0.0, 0.0, 0.0])
        self.Kd_ang_pos = np.array([0.0, 0.0, 0.0])
        self.lim_ang_pos = np.array([5.0, 5.0, 5.0])

        # outputs angular position
        self.Kp_lin_vel = np.array([0.13, 0.13])
        self.Ki_lin_vel = np.array([0.0015, 0.0015])
        self.Kd_lin_vel = np.array([0.003, 0.003])
        self.lim_lin_vel = np.array([0.6, 0.6])

        # height controllers
        z_pos_PID = PID(5.0, 0.0, 0.0, 10.0, self.ctrl_period)
        z_vel_PID = PID(1.5, 0.1, 0.25, 1.0, self.ctrl_period)
        self.z_PIDs = [z_vel_PID, z_pos_PID]
        self.PIDs = []

        self.reset()

        """ CAMERA """
        self.proj_mat = self.p.computeProjectionMatrixFOV(
            fov=camera_FOV, aspect=1.0, nearVal=0.1, farVal=255.0
        )
        self.rgbImg = self.depthImg = self.segImg = None
        self.rel_cam_Hz = int(sim_hz / cam_hz)
        self.camera_FOV = camera_FOV
        self.frame_size = np.array(frame_size)
        self.seg_size = (self.frame_size / seg_ratio).astype(np.int)

        # form the array for inverse projection later
        seg_centre = (self.seg_size + 1) / 2
        rpp = (camera_FOV / 180 * math.pi) / self.seg_size
        xspace = np.arange(self.seg_size[0], 0, -1)
        yspace = np.arange(self.seg_size[1], 0, -1)
        a_array = np.stack(np.meshgrid(xspace, yspace), axis=-1) - seg_centre
        a_array *= rpp
        a_array[:, :, 1] += math.pi / 4
        a_array = a_array.reshape(-1, 2)

        y = np.sin(a_array[:, 1]) / abs(np.cos(a_array[:, 1]))
        x = np.tan(a_array[:, 0]) / abs(np.cos(a_array[:, 1]))

        self.inv_proj = np.stack((x, y), axis=-1)

    def reset(self):
        self.set_mode(0)
        self.steps = 0
        self.state = np.zeros((4, 3))
        self.setpoint = np.zeros((4))
        self.rpm = np.zeros((4))
        self.pwm = np.zeros((4))

        for PID in self.PIDs:
            PID.reset()

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)

    def rpm2forces(self, rpm):
        """maps rpm to individual motor forces and torques"""
        rpm = np.expand_dims(rpm, axis=1)
        thrust = rpm * self.thr_coeff
        torque = rpm * self.tor_coeff * self.tor_dir

        # add some random noise to the motor outputs
        thrust += np.random.randn(*thrust.shape) * self.noise_ratio * thrust
        torque += np.random.randn(*torque.shape) * self.noise_ratio * torque

        for idx, thr, tor in zip(self.motor_id, thrust, torque):
            self.p.applyExternalForce(
                self.Id, idx, thr, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.Id, idx, tor, self.p.LINK_FRAME)

    def pwm2rpm(self, pwm):
        """model the motor using first order ODE, y' = T/tau * (setpoint - y)"""
        self.rpm += (self.sim_period / self.motor_tau) * (self.max_rpm * pwm - self.rpm)

        return self.rpm

    def cmd2pwm(self, cmd):
        """maps angular torque commands to motor rpms"""
        pwm = np.matmul(self.motor_map, cmd)

        min = np.min(pwm)
        max = np.max(pwm)

        # deal with motor saturations
        if min < 0.0:
            pwm = pwm - min
        if max > 1.0:
            pwm = pwm / max

        return pwm

    def update_state(self):
        """ang_vel, ang_pos, lin_vel, lin_pos"""
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

    def set_mode(self, mode):
        """
        sets the flight mode:
            0 - vp, vq, vr, vz  :  \n
            1 - p, q, r, vz     :  \n
            2 - vp, vq, vr, z   :  \n
            3 - p, q, r, z      :  \n
            4 - u, v, vr, z     :  \n
        """

        self.mode = mode
        if mode == 0 or mode == 2:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID]
        elif mode == 1 or mode == 3:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos,
                self.Ki_ang_pos,
                self.Kd_ang_pos,
                self.lim_ang_pos,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID]
        elif mode == 4:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos[:2],
                self.Ki_ang_pos[:2],
                self.Kd_ang_pos[:2],
                self.lim_ang_pos[:2],
                self.ctrl_period,
            )
            lin_vel_PID = PID(
                self.Kp_lin_vel,
                self.Ki_lin_vel,
                self.Kd_lin_vel,
                self.lim_lin_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID, lin_vel_PID]

    def update_control(self):
        """runs through PID controllers"""
        output = None
        # angle controllers
        if self.mode == 0 or self.mode == 2:
            output = self.PIDs[0].step(self.state[0], self.setpoint[:3])
        elif self.mode == 1 or self.mode == 3:
            output = self.PIDs[1].step(self.state[1], self.setpoint[:3])
            output = self.PIDs[0].step(self.state[0], output)
        elif self.mode == 4:
            output = self.PIDs[2].step(self.state[2][:2], self.setpoint[:2])
            output = np.array([-output[1], output[0]])
            output = self.PIDs[1].step(self.state[1][:2], output)
            output = self.PIDs[0].step(
                self.state[0], np.array([*output, self.setpoint[2]])
            )

        z_output = None
        # height controllers
        if self.mode == 0 or self.mode == 1:
            z_output = self.z_PIDs[0].step(self.state[2][-1], self.setpoint[-1])
            z_output = np.clip(z_output, 0, 1)
        elif self.mode == 2 or self.mode == 3 or self.mode == 4:
            z_output = self.z_PIDs[1].step(self.state[3][-1], self.setpoint[-1])
            z_output = self.z_PIDs[0].step(self.state[2][-1], z_output)
            z_output = np.clip(z_output, 0, 1)

        # mix the commands to generate the pwm signal
        self.pwm = self.cmd2pwm(np.array([*output, z_output]))

    def update_forces(self):
        self.rpm2forces(self.pwm2rpm(self.pwm))

    def update(self):
        """
        updates state and control
        """
        # update states according to sim rate
        self.update_state()

        # update control only when needed
        self.steps += 1
        if self.steps % self.ctrl_update_ratio == 0:
            self.update_control()

        # update motor outputs constantly
        self.update_forces()

    @property
    def view_mat(self):
        # get the state of the camera on the robot
        camera_state = self.p.getLinkState(self.Id, 4)

        # pose and rot
        position = camera_state[0]

        # simulate gimballed camera, looking downwards at 45 degrees
        rotation = None
        if True:
            rotation = np.array(self.p.getEulerFromQuaternion(camera_state[1]))
            rotation[0] = 0.0
            rotation[1] = 45 / 180 * math.pi
            # rotation[1] = (90 - 0.5 * self.camera_FOV) / 180 * math.pi
            rotation = np.array(self.p.getQuaternionFromEuler(rotation))
            rotation = np.array(self.p.getMatrixFromQuaternion(rotation)).reshape(3, 3)
        else:
            rotation = np.array(
                self.p.getMatrixFromQuaternion(camera_state[1])
            ).reshape(3, 3)

        # compute camera up vector using rot_mat
        up_vector = np.matmul(rotation, np.array([0.0, 0.0, 1.0]))

        # target position is 1000 units ahead of camera relative to the current camera pos
        target = np.dot(rotation, np.array([1000, 0, 0])) + np.array(position)

        return self.p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up_vector,
        )

    def capture_image(self):
        _, _, self.rgbImg, self.depthImg, self.segImg = self.p.getCameraImage(
            width=self.frame_size[1],
            height=self.frame_size[0],
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
            # lightAmbientCoeff=1.,
            # lightDiffuseCoeff=1.,
            # lightSpecularCoeff=1.
        )
        self.segImg = cv2.resize(
            self.segImg,
            (self.seg_size[0], self.seg_size[1]),
            interpolation=cv2.INTER_NEAREST,
        )
        # self.rgbImg = cv2.GaussianBlur(self.rgbImg, (3, 3), 0)
