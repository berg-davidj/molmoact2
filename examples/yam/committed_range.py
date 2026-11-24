"""Committed rows a GR00T Worker is conditioned on.

The Holesaw GR00T checkpoint normalizes the gripper within about [0, 0.96]. A Worker without the platform
clamp (PR #2661) rejects committed gripper rows outside it, and the YAM rollout commits 1.0 for a fully open
gripper. The Worker is conditioned on the clamped rows; the chunk handed back to the robot keeps the rows the
robot executes.
"""

import numpy as np

GR00T_COMMITTED_GRIPPER_RANGE = (0.0, 0.95)


def gr00t_conditioning_prefix(prefix):
    """The committed rows to send: the gripper clamped into the checkpoint's range, the joints untouched."""
    conditioned = np.array(prefix, dtype=np.float32, copy=True)
    conditioned[:, 6] = np.clip(conditioned[:, 6], *GR00T_COMMITTED_GRIPPER_RANGE)
    return conditioned
