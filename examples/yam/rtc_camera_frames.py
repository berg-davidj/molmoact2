"""Select a fresh set of frames for an RTC observation."""

import time
from collections import deque
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from itertools import product
from threading import Condition

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    ready_at: float
    captured_at: float | None


def frame_times(frames: Collection[CameraFrame]) -> list[float]:
    """Use one clock source for the whole observation."""
    driver_times = [frame.captured_at for frame in frames if frame.captured_at is not None]
    return driver_times if len(driver_times) == len(frames) else [frame.ready_at for frame in frames]


class RtcCameraFrames:
    def __init__(self, roles: Iterable[str], frame_period: float, *, skew_limit: float | None = None):
        self.frame_period = frame_period
        self.skew_limit = frame_period / 2 if skew_limit is None else skew_limit
        self._frames = {role: deque(maxlen=3) for role in roles}
        self._updated = Condition()

    def publish(self, role: str, image: np.ndarray, ready_at: float, captured_at: float | None) -> None:
        with self._updated:
            self._frames[role].append(CameraFrame(image, ready_at, captured_at))
            self._updated.notify()

    def take(self, max_wait: float) -> dict[str, CameraFrame] | None:
        """Wait for a newly available, aligned set within the caller's tick budget."""
        deadline = time.monotonic() + min(max_wait, self.frame_period)
        with self._updated:
            while True:
                now = time.monotonic()
                fresh = []
                fallback = []
                for frames in product(*self._frames.values()):
                    oldest_ready = min(frame.ready_at for frame in frames)
                    if now - oldest_ready > 3 * self.frame_period:
                        continue
                    timestamps = frame_times(frames)
                    oldest = min(timestamps)
                    skew = max(timestamps) - oldest
                    aligned = skew <= self.skew_limit
                    fallback.append((aligned, oldest, -skew, frames))
                    newest_ready = max(frame.ready_at for frame in frames)
                    if aligned and now - newest_ready <= self.frame_period / 8:
                        fresh.append((skew, -oldest, frames))
                if fresh:
                    return dict(zip(self._frames, min(fresh, key=lambda item: item[:2])[-1]))
                if now >= deadline:
                    if fallback:
                        return dict(zip(self._frames, max(fallback, key=lambda item: item[:3])[-1]))
                    return None
                self._updated.wait(deadline - now)
