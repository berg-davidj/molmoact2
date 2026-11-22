import time

import numpy as np

from frame_capture_time import session_capture_time
from gello_min.env import RobotEnv


class FakeRobot:
    def num_dofs(self):
        return 7

    def get_observations(self):
        zeros = np.zeros(7, dtype=np.float32)
        return {"joint_positions": zeros, "joint_velocities": zeros, "ee_pos_quat": zeros,
                "gripper_position": np.zeros(1, dtype=np.float32)}


class FakeCamera:
    def __init__(self, captured_at=None):
        self._captured_at = captured_at
        self.last_read_timestamp = None

    def read(self):
        self.last_read_timestamp = self._captured_at
        return np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 1), dtype=np.uint16)


class FakeCameraClient:
    def __init__(self, timestamps):
        self.last_timestamps = {}
        self._timestamps = timestamps

    def get_obs(self):
        self.last_timestamps = dict(self._timestamps)
        return {name: np.zeros((2, 2, 3), dtype=np.uint8) for name in self._timestamps}


class FakeClock:
    def __init__(self, now):
        self._now = now

    def now(self):
        return self._now


def test_get_obs_keeps_the_capture_time_of_each_frame_it_returned():
    env = RobotEnv(FakeRobot(), camera_dict={"front_camera": FakeCamera(100.0), "left_camera": FakeCamera(100.5),
                                             "untimed_camera": FakeCamera(None)})
    obs = env.get_obs()
    assert env.last_frame_timestamps == {"front_camera": 100.0, "left_camera": 100.5}
    assert {"front_camera_rgb", "left_camera_rgb", "untimed_camera_rgb", "joint_positions"} <= set(obs)
    assert not any("timestamp" in key for key in obs)  # what get_obs() returns is unchanged

    served = RobotEnv(FakeRobot(), camera_client=FakeCameraClient({"front_camera": 7.0}))
    served.get_obs()
    assert served.last_frame_timestamps == {"front_camera": 7.0}


def test_capture_time_is_the_oldest_frame_on_the_session_clock():
    env = RobotEnv(FakeRobot(), camera_dict={"front_camera": FakeCamera(time.time() - 0.040),
                                             "left_camera": FakeCamera(time.time() - 0.010)})
    env.get_obs()
    captured_at = session_capture_time(env, FakeClock(5.0))
    assert abs(captured_at - (5.0 - 0.040)) < 0.005  # 40 ms before "now" on the Session clock

    untimed = RobotEnv(FakeRobot(), camera_dict={"front_camera": FakeCamera(None)})
    untimed.get_obs()
    assert session_capture_time(untimed, FakeClock(5.0)) is None
