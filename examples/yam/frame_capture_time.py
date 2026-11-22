"""When the camera frames of an observation were captured, on the Session clock.

The camera drivers stamp each frame with ``time.time()`` as they read it from the device, and
``RobotEnv.get_obs()`` keeps the stamps of the frames it returned in ``last_frame_timestamps``.
The Orchestral Session clock is monotonic, so the stamp is carried over as an age: how long ago
the frame was captured, taken off the Session's current time. Call this right after
``get_obs()``, so the two clocks are read at the same moment.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def session_capture_time(env: Any, clock: Any) -> Optional[float]:
    """Session time at which the oldest frame of the last ``env.get_obs()`` was captured.

    The oldest frame is used because an observation is only as fresh as its stalest camera.
    None when the cameras reported no capture time, or when the frame's stamp is ahead of
    the wall clock (the clock was set back since the capture): no figure rather than a wrong one.
    """
    timestamps = getattr(env, "last_frame_timestamps", None)
    if not timestamps:
        return None
    age = time.time() - min(timestamps.values())
    if age < 0:
        return None
    return clock.now() - age
