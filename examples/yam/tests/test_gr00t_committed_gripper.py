"""The GR00T bridge conditions the Worker on a gripper inside the checkpoint's range and keeps the executed rows."""

import numpy as np

from orchestral_session_bridge import GR00T_COMMITTED_GRIPPER_RANGE, gr00t_conditioning_prefix


def test_open_and_closed_gripper_are_clamped_for_conditioning_only():
    rows = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.02],
                     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5]], dtype=np.float32)
    original = rows.copy()
    conditioned = gr00t_conditioning_prefix(rows)
    lo, hi = GR00T_COMMITTED_GRIPPER_RANGE
    assert conditioned[:, 6].tolist() == [float(np.float32(hi)), lo, 0.5]
    assert np.array_equal(conditioned[:, :6], rows[:, :6])
    assert np.array_equal(rows, original)
