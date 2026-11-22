import time
from typing import Any, Dict, Optional

import numpy as np

from gello_min.camera import CameraDriver
from gello_min.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        camera_client: Optional[Any] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        # When set, get_obs() pulls images from the camera server over ZMQ
        # instead of opening RealSense devices in-process. camera_dict is
        # then ignored. See gello/cameras/camera_client.py.
        self._camera_client = camera_client
        # Capture time per camera of the frames the last get_obs() returned, on time.time()
        # and on time.monotonic(); None for a camera that reported none (frame_capture_time.py).
        self.last_frame_timestamps: Dict[str, Optional[float]] = {}
        self.last_frame_monotonic: Dict[str, Optional[float]] = {}

    # dynamic offset is used in data collection to make sure the same
    # starting position between each episode.
        self._dynamic_offset = np.zeros(self._robot.num_dofs())
        self._original_offset = np.zeros(self._robot.num_dofs())

    def set_original_offset(self, gello_joints: np.ndarray) -> None:
        self._original_offset = gello_joints - self._robot.get_joint_state()

    def set_dynamic_offset(self, gello_joints: np.ndarray) -> None:
        self._dynamic_offset = gello_joints - self._robot.get_joint_state() - self._original_offset

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray, reset: Optional[bool] = False) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        self.step_command_only(joints, reset=reset)
        return self.get_obs()

    def step_command_only(
        self, joints: np.ndarray, reset: Optional[bool] = False, wait: bool = True
    ) -> None:
        """Command the robot + sleep on the control rate. Does NOT read cameras.

        Use this inside tight inner loops (e.g. interpolated sub-steps of an
        action) so each tick doesn't pay for a full ``get_obs()``. Call
        ``get_obs()`` once after the loop when you actually need the obs.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)

        if reset:
            self._robot.command_joint_state(joints)
        else:
            self._robot.command_joint_state(joints - self._dynamic_offset)
        if wait:
            self._rate.sleep()

    def wait_for_control_tick(self) -> None:
        """Wait for the next outer control tick."""
        self._rate.sleep()

    def get_robot_state(self) -> Dict[str, Any]:
        """Robot-only observations (joint positions/velocities, EE pose, gripper).

        Same fields as ``get_obs()`` minus the ``*_rgb`` camera images. Use this
        when you only need joints (e.g. computing an interpolation target).
        """
        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        return {
            "joint_positions": robot_obs["joint_positions"],
            "joint_velocities": robot_obs["joint_velocities"],
            "ee_pos_quat": robot_obs["ee_pos_quat"],
            "gripper_position": robot_obs["gripper_position"],
        }

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations: Dict[str, Any] = {}
        timestamps: Dict[str, Optional[float]] = {}
        monotonic: Dict[str, Optional[float]] = {}
        if self._camera_client is not None:
            frames = self._camera_client.get_obs()
            served = getattr(self._camera_client, "last_timestamps", None) or {}
            served_monotonic = getattr(self._camera_client, "last_monotonic", None) or {}
            for name, image in frames.items():
                observations[f"{name}_rgb"] = image
                timestamps[name] = served.get(name)
                monotonic[name] = served_monotonic.get(name)
        else:
            for name, camera in self._camera_dict.items():
                # Other threads (previews, recorders) read the same cameras, so the capture
                # time has to come back with the frame, not from a shared attribute.
                read_with_capture_time = getattr(camera, "read_with_capture_time", None)
                if read_with_capture_time is not None:
                    image, _depth, timestamp, mono = read_with_capture_time()
                else:
                    image, _depth = camera.read()
                    timestamp = mono = None
                observations[f"{name}_rgb"] = image
                timestamps[name] = float(timestamp) if timestamp else None
                monotonic[name] = float(mono) if mono else None
        self.last_frame_timestamps = timestamps
        self.last_frame_monotonic = monotonic

        observations.update(self.get_robot_state())
        return observations

    def close(self) -> None:
        """Release physical robot control and camera handles deterministically.

        The robot is closed first so its non-daemon control thread stops and
        motors are disabled through the driver's regular shutdown path. Camera
        capture threads are then stopped, which releases V4L2 devices for a
        camera-only viewer or the next session.
        """
        first_error = None
        close_robot = getattr(self._robot, "close", None)
        if callable(close_robot):
            try:
                close_robot()
            except Exception as exc:  # noqa: BLE001 -- still release cameras
                first_error = exc

        if self._camera_client is not None:
            try:
                self._camera_client.close()
            except Exception as exc:  # noqa: BLE001 -- best-effort teardown
                if first_error is None:
                    first_error = exc
            self._camera_client = None

        for camera in self._camera_dict.values():
            close_camera = getattr(camera, "close", None)
            if not callable(close_camera):
                continue
            try:
                close_camera()
            except Exception as exc:  # noqa: BLE001 -- close remaining cameras
                if first_error is None:
                    first_error = exc
        self._camera_dict = {}

        if first_error is not None:
            raise first_error


def main() -> None:
    pass


if __name__ == "__main__":
    main()
