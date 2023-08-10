import threading

import numpy as np
from inputs import get_gamepad


class XboxController(object):
    MAX_JOY_VAL = 2**15

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller, args=()
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):
        # yaw
        left_x = self.LeftJoystickX
        # pitch
        left_y = self.LeftJoystickY
        # roll
        right_x = self.RightJoystickX
        # thrust
        right_y = self.RightJoystickY
        return np.array([right_x, -left_y, left_x, -right_y])

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_Y":
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL
                elif event.code == "ABS_X":
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL
                elif event.code == "ABS_RY":
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL
                elif event.code == "ABS_RX":
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL


if __name__ == "__main__":
    joy = XboxController()
    while True:
        print(joy.read())
