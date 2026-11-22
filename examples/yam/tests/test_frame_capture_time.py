import numpy as np

import frame_capture_time
from frame_capture_time import session_capture_time
from gello_min.env import RobotEnv

FRAME = np.zeros((2, 2, 3), dtype=np.uint8)
DEPTH = np.zeros((2, 2, 1), dtype=np.uint16)


class FakeRobot:
    def num_dofs(self):
        return 7

    def get_observations(self):
        zeros = np.zeros(7, dtype=np.float32)
        return {"joint_positions": zeros, "joint_velocities": zeros, "ee_pos_quat": zeros,
                "gripper_position": np.zeros(1, dtype=np.float32)}


class TimedCamera:
    """A camera that returns each frame's capture time with the frame."""

    def __init__(self, captured_at):
        self._captured_at = captured_at

    def read(self):
        return FRAME, DEPTH

    def read_with_timestamp(self):
        return FRAME, DEPTH, self._captured_at


class UntimedCamera:
    def read(self):
        return FRAME, DEPTH


class FakeCameraClient:
    def __init__(self, timestamps):
        self.last_timestamps = {}
        self._timestamps = timestamps

    def get_obs(self):
        self.last_timestamps = dict(self._timestamps)
        return {name: FRAME for name in self._timestamps}


class FakeClock:
    def __init__(self, now):
        self._now = now

    def now(self):
        return self._now


def test_get_obs_keeps_the_capture_time_of_each_frame_it_returned():
    env = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(100.0), "left_camera": TimedCamera(100.5),
                                             "untimed_camera": UntimedCamera()})
    obs = env.get_obs()
    assert env.last_frame_timestamps == {"front_camera": 100.0, "left_camera": 100.5}
    assert {"front_camera_rgb", "left_camera_rgb", "untimed_camera_rgb", "joint_positions"} <= set(obs)
    assert not any("timestamp" in key for key in obs)  # what get_obs() returns is unchanged

    served = RobotEnv(FakeRobot(), camera_client=FakeCameraClient({"front_camera": 7.0}))
    served.get_obs()
    assert served.last_frame_timestamps == {"front_camera": 7.0}


def test_capture_time_is_the_oldest_frame_on_the_session_clock(monkeypatch):
    monkeypatch.setattr(frame_capture_time.time, "time", lambda: 1000.0)
    env = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(1000.0 - 0.040),
                                             "left_camera": TimedCamera(1000.0 - 0.010)})
    env.get_obs()
    # The oldest frame is 40 ms old, so it was captured 40 ms before "now" on the Session clock.
    assert abs(session_capture_time(env, FakeClock(5.0)) - 4.960) < 1e-9

    untimed = RobotEnv(FakeRobot(), camera_dict={"front_camera": UntimedCamera()})
    untimed.get_obs()
    assert session_capture_time(untimed, FakeClock(5.0)) is None

    ahead = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(1000.5)})
    ahead.get_obs()
    assert session_capture_time(ahead, FakeClock(5.0)) is None  # wall clock set back: no figure


def _v4l2_camera_with_cached_frame(timestamp):
    import threading

    from gello_min.v4l2_camera import V4L2Camera

    camera = object.__new__(V4L2Camera)
    camera.device = "fake-v4l2"
    camera._frame_ready = threading.Event()
    camera._frame_ready.set()
    camera._frame_lock = threading.Lock()
    camera._latest_rgb = np.full((3, 4, 3), 17, dtype=np.uint8)
    camera._latest_depth = np.zeros((3, 4, 1), dtype=np.uint16)
    camera._latest_frame_timestamp = timestamp
    camera._last_capture_error = None
    camera._read_wait_timeout_sec = 0.1
    camera._max_frame_age_sec = 1.0
    return camera


def _realsense_camera_with_cached_frame(timestamp):
    import threading

    from gello_min.realsense_camera import RealSenseCamera

    camera = object.__new__(RealSenseCamera)
    camera._frame_ready = threading.Event()
    camera._frame_ready.set()
    camera._frame_lock = threading.Lock()
    camera._latest_color_image = np.full((3, 4, 3), 17, dtype=np.uint8)
    camera._latest_depth_image = np.zeros((3, 4), dtype=np.uint16)
    camera._latest_frame_timestamp = timestamp
    camera._last_capture_error = None
    camera._read_wait_timeout_sec = 0.1
    camera._max_frame_age_sec = 1.0
    camera._flip = False
    return camera


def test_camera_drivers_return_the_capture_time_with_the_frame():
    import time

    for build in (_v4l2_camera_with_cached_frame, _realsense_camera_with_cached_frame):
        stored_at = time.time() - 0.020
        camera = build(stored_at)
        rgb, depth, timestamp = camera.read_with_timestamp()
        assert timestamp == stored_at
        plain_rgb, plain_depth = camera.read()  # read() still returns the same two arrays
        np.testing.assert_array_equal(plain_rgb, rgb)
        assert plain_depth.shape == depth.shape == (3, 4, 1)

        env = RobotEnv(FakeRobot(), camera_dict={"front_camera": camera})
        env.get_obs()
        assert env.last_frame_timestamps == {"front_camera": stored_at}
