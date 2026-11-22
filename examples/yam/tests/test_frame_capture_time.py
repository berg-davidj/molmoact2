import time

import numpy as np

import frame_capture_time
from frame_capture_time import session_capture_time
from gello_min.env import RobotEnv
from rtc_camera_frames import RtcCameraFrames, frame_times

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
    """A camera that returns each frame's capture time, on both clocks, with the frame."""

    def __init__(self, captured_at, monotonic=None):
        self._captured_at = captured_at
        self._monotonic = monotonic

    def read(self):
        return FRAME, DEPTH

    def read_with_capture_time(self):
        return FRAME, DEPTH, self._captured_at, self._monotonic


class UntimedCamera:
    def read(self):
        return FRAME, DEPTH


class FakeCameraClient:
    def __init__(self, timestamps):
        self.last_timestamps = {}
        self.last_monotonic = {}
        self._timestamps = timestamps

    def get_obs(self):
        self.last_timestamps = dict(self._timestamps)
        self.last_monotonic = {name: None for name in self._timestamps}
        return {name: FRAME for name in self._timestamps}


class FakeClock:
    """A Session clock whose source is not time.monotonic(): now() is fixed."""

    def __init__(self, now):
        self._now = now

    def now(self):
        return self._now

    def to_source(self, t):
        return t - 1e6


class MonotonicClock:
    """A Session clock on time.monotonic(), like orchestral's default SessionClock."""

    def __init__(self, opened_at):
        self.opened_at = opened_at

    def now(self):
        return time.monotonic() - self.opened_at

    def to_source(self, t):
        return t + self.opened_at

    def from_source(self, t):
        return t - self.opened_at


def test_get_obs_keeps_the_capture_time_of_each_frame_it_returned():
    env = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(100.0), "left_camera": TimedCamera(100.5),
                                             "untimed_camera": UntimedCamera()})
    obs = env.get_obs()
    assert env.last_frame_timestamps == {"front_camera": 100.0, "left_camera": 100.5, "untimed_camera": None}
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

    # On a Session clock that reads time.monotonic(), the monotonic stamps are used as they are.
    on_monotonic = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(1.0, monotonic=12.5),
                                                      "left_camera": TimedCamera(1.0, monotonic=12.75)})
    on_monotonic.get_obs()
    clock = MonotonicClock(opened_at=10.0)
    assert session_capture_time(on_monotonic, clock) == 2.5

    # One camera without a capture time: no figure, rather than the other cameras' younger one.
    partly = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(1000.0 - 0.040),
                                                "untimed_camera": UntimedCamera()})
    partly.get_obs()
    assert session_capture_time(partly, FakeClock(5.0)) is None

    untimed = RobotEnv(FakeRobot(), camera_dict={"front_camera": UntimedCamera()})
    untimed.get_obs()
    assert session_capture_time(untimed, FakeClock(5.0)) is None

    ahead = RobotEnv(FakeRobot(), camera_dict={"front_camera": TimedCamera(1000.5)})
    ahead.get_obs()
    assert session_capture_time(ahead, FakeClock(5.0)) is None  # wall clock set back: no figure


def test_capture_time_uses_only_prediction_cameras(monkeypatch):
    monkeypatch.setattr(frame_capture_time.time, "time", lambda: 1000.0)
    env = RobotEnv(FakeRobot(), camera_dict={
        "front_camera": TimedCamera(999.98, monotonic=12.5),
        "left_camera": TimedCamera(999.99, monotonic=12.75),
        "right_camera": TimedCamera(999.90, monotonic=12.0),
        "untimed_camera": UntimedCamera(),
    })
    env.get_obs()
    selected = ("front_camera", "left_camera")
    clock = MonotonicClock(opened_at=10.0)
    assert session_capture_time(env, clock, camera_names=selected) == 2.5
    assert abs(session_capture_time(env, FakeClock(5.0), camera_names=selected) - 4.98) < 1e-9
    assert session_capture_time(env, clock, camera_names=("left_camera",)) == 2.75
    assert session_capture_time(env, clock, camera_names=("left_camera", "missing")) is None


def test_rtc_pairs_driver_timestamps_even_when_delivery_order_differs():
    now = time.monotonic()
    frames = RtcCameraFrames(("top", "left"), 1 / 30)
    frames.publish("left", FRAME, now - 0.025, now - 0.033)
    frames.publish("left", np.ones_like(FRAME), now - 0.001, now - 0.008)
    frames.publish("top", FRAME, now, now - 0.032)

    selected = frames.take(0)
    assert selected["left"].image is FRAME
    assert np.isclose(np.ptp(frame_times(selected.values())), 0.001)


def test_rtc_pairing_uses_one_clock_and_rejects_incomplete_or_stale_sets():
    now = time.monotonic()
    frames = RtcCameraFrames(("top", "left"), 1 / 30)
    frames.publish("top", FRAME, now - 0.003, now - 0.034)
    assert frames.take(0) is None
    frames.publish("left", FRAME, now - 0.002, None)
    assert np.isclose(np.ptp(frame_times(frames.take(0).values())), 0.001)

    stale = RtcCameraFrames(("top",), 1 / 30)
    stale.publish("top", FRAME, now - 1, now - 1)
    assert stale.take(0) is None


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
    camera._latest_frame_monotonic = 42.0
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
    camera._latest_frame_monotonic = 42.0
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
        rgb, depth, timestamp, monotonic = camera.read_with_capture_time()
        assert (timestamp, monotonic) == (stored_at, 42.0)
        plain_rgb, plain_depth = camera.read()  # read() still returns the same two arrays
        np.testing.assert_array_equal(plain_rgb, rgb)
        assert plain_depth.shape == depth.shape == (3, 4, 1)

        env = RobotEnv(FakeRobot(), camera_dict={"front_camera": camera})
        env.get_obs()
        assert env.last_frame_timestamps == {"front_camera": stored_at}
        assert env.last_frame_monotonic == {"front_camera": 42.0}
