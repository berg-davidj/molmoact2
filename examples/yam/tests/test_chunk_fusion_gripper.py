"""The gripper is never blended, but a fused mode lets the newest chunk replace the queued gripper plan at once.

On the robot (DW05, crossfade) that released the gripper early: each chunk's gripper transition time jitters, so the first
chunk whose guess lands early fires the release. `hold_gripper_rows` keeps the queued plan's gripper for the next N slots.
"""
import numpy as np
import pytest

from chunk_fusion import ChunkPlan


def ramp(start, rows=16, step=0.01, gripper=0.0):
    chunk = np.empty((rows, 7), dtype=np.float32)
    chunk[:, :6] = (start + step * np.arange(rows))[:, None]
    chunk[:, 6] = gripper
    return chunk


def drain(plan):
    rows = []
    while plan:
        rows.append(plan.pop())
    return rows


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_hold_gripper_rows_keeps_the_queued_gripper_but_still_fuses_the_joints(mode):
    # The queued plan keeps the gripper closed (0.0); a new chunk that jittered early says open (1.0) from its first row.
    plan = ChunkPlan(mode, blend_rows=4, hold_gripper_rows=4)
    plan.add(ramp(0.0, gripper=0.0), 0)
    for _ in range(2):
        plan.pop()
    plan.add(ramp(0.07, gripper=1.0), 2)  # the joints disagree with the queued plan, and so does the gripper
    rows = np.array([row for _, row in drain(plan)])
    assert len(rows) >= 8
    # The next 4 slots keep the queued gripper: one chunk, not the first to land early, decides the release ...
    np.testing.assert_array_equal(rows[:4, 6], 0.0)
    # ... and later slots follow the newest chunk.
    np.testing.assert_array_equal(rows[4:, 6], 1.0)


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_the_joints_still_fuse_with_the_gripper_held(mode):
    held = ChunkPlan(mode, blend_rows=4, hold_gripper_rows=4)
    free = ChunkPlan(mode, blend_rows=4)
    for plan in (held, free):
        plan.add(ramp(0.0, gripper=0.0), 0)
        for _ in range(2):
            plan.pop()
        plan.add(ramp(0.07, gripper=1.0), 2)
    np.testing.assert_allclose(np.array([r for _, r in drain(held)])[:, :6], np.array([r for _, r in drain(free)])[:, :6])


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_the_gripper_follows_the_newest_chunk_at_once_by_default(mode):
    plan = ChunkPlan(mode, blend_rows=4)
    plan.add(ramp(0.0, gripper=0.0), 0)
    for _ in range(2):
        plan.pop()
    plan.add(ramp(0.07, gripper=1.0), 2)
    assert plan.pop()[1][6] == 1.0  # the existing behaviour, unchanged for every other user of the fusion modes


def test_splice_is_unaffected_by_the_hold():
    plan = ChunkPlan("splice", hold_gripper_rows=4)
    plan.add(ramp(0.0, gripper=0.0), 0)
    for _ in range(8):
        plan.pop()
    plan.add(ramp(0.08, gripper=1.0), 8)  # splice keeps the 8 queued rows and appends after them
    grippers = [row[6] for _, row in drain(plan)]
    assert grippers == [0.0] * 8 + [1.0] * 8


def test_hold_gripper_rows_must_not_be_negative():
    with pytest.raises(ValueError, match="hold_gripper_rows"):
        ChunkPlan("crossfade", hold_gripper_rows=-1)
