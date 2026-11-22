"""Join successive action chunks into one stream of per-tick commands.

A chunk's row i is the target for command slot `first_slot + i`, where `first_slot` is the
slot the rollout had reached when it took the observation for that chunk. Rows for slots
that have already been commanded are stale and are discarded. The modes differ in what
they do with the slots that both the queued plan and the new chunk cover:

- splice     keep the queued rows and append the new chunk after them (no blending).
- crossfade  ramp linearly from the queued rows to the new chunk over `blend_rows` slots.
- ensemble   ACT temporal ensembling: average every chunk's prediction for a slot,
             weighting older predictions more (Zhao et al. 2023).
- rail       VLA-RAIL (arXiv 2512.24673): replace each chunk with its per-joint cubic
             least-squares fit, then bridge from the executing trajectory to the new one
             with two quintics that match position, velocity and acceleration.

Only the joint columns are fused. The last column is the gripper aperture, where an
averaged or fitted value would delay a grasp, so it always follows the newest chunk.
"""
from collections import deque

import numpy as np

MODES = ("splice", "crossfade", "ensemble", "rail")
# ACT's weight for a prediction that is i chunks older than the oldest one is exp(-m * i).
ENSEMBLE_DECAY = 0.01


def _quintic(p0, v0, a0, p1, v1, a1, duration):
    """Coefficients (lowest power first) of the quintic joining two motion states."""
    t = duration
    return np.stack([
        p0,
        v0,
        a0 / 2,
        (20 * (p1 - p0) - (8 * v1 + 12 * v0) * t - (3 * a0 - a1) * t**2) / (2 * t**3),
        (30 * (p0 - p1) + (14 * v1 + 16 * v0) * t + (3 * a0 - 2 * a1) * t**2) / (2 * t**4),
        (12 * (p1 - p0) - 6 * (v1 + v0) * t - (a0 - a1) * t**2) / (2 * t**5),
    ])


def _rail_bridge(start, end, rows):
    """VLA-RAIL's dual quintic from `start` at tick 0 to `end` at tick `rows`.

    Each motion state is (position, velocity, acceleration). The two quintics meet halfway
    at the mean position and velocity with zero acceleration. Returns ticks 1..rows-1.
    """
    half = rows / 2
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, np.zeros_like(start[0]))
    left = _quintic(*start, *middle, half)
    right = _quintic(*middle, *end, half)
    points = []
    for tick in range(1, rows):
        coefficients, t = (left, tick) if tick < half else (right, tick - half)
        points.append(sum(c * t**power for power, c in enumerate(coefficients)))
    return points


class ChunkPlan:
    """The queue of rows still to be commanded, fused from the chunks added to it."""

    def __init__(self, mode="splice", *, blend_rows=4):
        if mode not in MODES:
            raise ValueError(f"Unknown chunk fusion mode {mode!r}; choose from {', '.join(MODES)}")
        if blend_rows < 1:
            raise ValueError("blend_rows must be at least 1")
        self.mode = mode
        self.blend_rows = blend_rows
        self.slot = 0  # the next command slot; equals the number of rows popped so far
        self._queue = deque()  # (row index in its chunk, row), one per upcoming slot
        self._votes = deque()  # ensemble only: every chunk's joint prediction per queued slot
        self._sent = deque(maxlen=2)  # joint columns of the last two rows popped

    @property
    def fuses(self):
        """Whether a new chunk should be added while rows from the last one remain."""
        return self.mode != "splice"

    def __len__(self):
        return len(self._queue)

    def pop(self):
        index, row = self._queue.popleft()
        if self._votes:
            self._votes.popleft()
        self.slot += 1
        self._sent.append(row[:-1])
        return index, row

    def committed_actions(self):
        """Snapshot the queued robot targets in execution order."""
        return np.stack([row for _, row in self._queue])

    def add(self, chunk, first_slot):
        """Fuse a chunk whose row 0 targets `first_slot`. Returns the first row index used,
        or None when every row is stale."""
        chunk = np.asarray(chunk, dtype=np.float32)
        stale = self.slot - first_slot
        if stale < 0:
            raise ValueError("a chunk cannot start after the slots already planned for it")
        if stale >= len(chunk):
            return None
        fuse = getattr(self, f"_{self.mode}")
        fuse(chunk, stale)
        return stale

    def _replace(self, rows, first_index):
        self._queue = deque(enumerate(rows, start=first_index))

    def _splice(self, chunk, stale):
        covered = len(self._queue)
        self._queue.extend(enumerate(chunk[stale + covered:], start=stale + covered))

    def _crossfade(self, chunk, stale):
        rows = chunk[stale:].copy()
        ramp = min(self.blend_rows, len(self._queue), len(rows))
        for k in range(ramp):
            weight = (k + 1) / (ramp + 1)
            rows[k, :-1] = (1 - weight) * self._queue[k][1][:-1] + weight * rows[k, :-1]
        self._replace(rows, stale)

    def _ensemble(self, chunk, stale):
        rows = chunk[stale:].copy()
        votes = deque()
        for k, row in enumerate(rows):
            earlier = self._votes[k] if k < len(self._votes) else []
            predictions = [*earlier, row[:-1].copy()]
            weights = np.exp(-ENSEMBLE_DECAY * np.arange(len(predictions)))
            row[:-1] = np.average(predictions, axis=0, weights=weights)
            votes.append(predictions)
        self._replace(rows, stale)
        self._votes = votes

    def _rail(self, chunk, stale):
        ticks = np.arange(len(chunk))
        fit = np.polyfit(ticks, chunk[:, :-1], min(3, len(chunk) - 1))
        smooth = chunk.copy()
        smooth[:, :-1] = np.stack([np.polyval(fit[:, j], ticks) for j in range(fit.shape[1])], axis=1)
        rows = smooth[stale:]
        bridge = min(self.blend_rows, len(rows))
        if self._sent and bridge >= 2:
            # Tick 0 is the last row commanded; tick `bridge` is the new chunk's row for
            # that slot, where the bridge hands over to the fitted chunk.
            position = self._sent[-1]
            velocity = acceleration = np.zeros_like(position)
            if self._queue and len(self._sent) == 2:
                upcoming = self._queue[0][1][:-1]
                velocity = (upcoming - self._sent[0]) / 2
                acceleration = upcoming - 2 * position + self._sent[0]
            elif self._queue:
                velocity = self._queue[0][1][:-1] - position
            end = stale + bridge - 1
            derivative = [np.polyder(np.poly1d(fit[:, j])) for j in range(fit.shape[1])]
            end_state = (
                smooth[end, :-1],
                np.array([d(end) for d in derivative]),
                np.array([np.polyder(d)(end) for d in derivative]),
            )
            for k, joints in enumerate(_rail_bridge((position, velocity, acceleration), end_state, bridge)):
                rows[k, :-1] = joints
        self._replace(rows, stale)
