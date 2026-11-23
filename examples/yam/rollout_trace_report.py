"""Score how smoothly a rollout ran, from the control trace in its Rerun recording.

    python examples/yam/rollout_trace_report.py                 # the newest recording
    python examples/yam/rollout_trace_report.py path/to/run.rrd

A smooth run has a tick rate equal to the control rate, a joint step at chunk changes
close to the step within a chunk, and few capped or held ticks.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun.experimental as rr_experimental

from rollout_trace import LATENCY_STAGES

RECORDINGS = Path("yam_eval_runs/rerun")
JOINTS = slice(0, 6)


def _unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def load_trace(path):
    """Rows of every /robot and /policy entity, in time order."""
    rows = defaultdict(list)
    for chunk in rr_experimental.RrdReader(path).stream().to_chunks():
        entity = str(chunk.entity_path)
        if chunk.is_static or not entity.startswith(("/robot", "/policy")):
            continue
        batch = chunk.to_record_batch()
        columns = {}
        for name in batch.schema.names:
            column = batch.column(name)
            if pa.types.is_duration(column.type) or pa.types.is_timestamp(column.type):
                column = column.cast(pa.int64())
            columns[name] = column.to_pylist()
        rows[entity].extend({name: values[i] for name, values in columns.items()} for i in range(batch.num_rows))
    return {entity: sorted(values, key=lambda row: row["elapsed"]) for entity, values in rows.items()}


def active_plan(trace):
    """Return the active arm's plan and its columns in the bimanual trace."""
    if "/robot/planned_right" in trace:
        return trace["/robot/planned_right"], slice(7, 14)
    return trace.get("/robot/planned_left", []), slice(0, 7)


def report(path):
    trace = load_trace(path)
    planned, arm = active_plan(trace)
    if len(planned) < 30:
        return f"{path.name}: no control trace (fewer than 30 planned rows)"
    plan = np.array([row["Scalars:scalars"] for row in planned])
    times = np.array([row["elapsed"] for row in planned]) / 1e9
    prediction = np.array([_unwrap(row["prediction_id"]) for row in planned])
    row_index = np.array([_unwrap(row["row_index"]) for row in planned])
    steps = np.abs(np.diff(plan, axis=0)[:, JOINTS])
    chunk_change = np.diff(prediction) != 0
    within = steps[~chunk_change].mean()
    at_change = steps[chunk_change].mean() if chunk_change.any() else float("nan")
    lines = [
        path.name,
        f"  tick rate          {1 / np.median(np.diff(times)):.1f} Hz over {len(plan)} commanded rows",
        f"  rows run per chunk {int(row_index[len(row_index) // 2:].min())}..{int(row_index.max())}",
        f"  joint step         {within:.4f} rad within a chunk, {at_change:.4f} rad at a chunk change "
        f"({at_change / within:.1f}x)",
        f"  plan roughness     {np.sqrt(np.mean(np.diff(plan[:, JOINTS], n=2, axis=0) ** 2)):.4f} rad "
        "(rms change in step between ticks)",
    ]
    commands = trace.get("/robot/command_sent", [])
    if len(commands) == len(plan):
        sent = np.array([row["Scalars:scalars"] for row in commands])[:, arm]
        capped = (np.abs(sent - plan)[:, JOINTS].max(axis=1) > 1e-4).mean()
        lines.append(f"  capped ticks       {100 * capped:.0f}% (the command differed from the plan)")
    else:
        lines.append("  capped ticks       n/a (shadow run: no commands sent)")
    holding = np.array([_unwrap(row["Scalars:scalars"]) for row in trace.get("/policy/holding", [])])
    if len(holding):
        lines.append(f"  hold ticks         {100 * holding.mean():.1f}%")
    latency = np.array([_unwrap(row["Scalars:scalars"]) for row in trace.get("/policy/latency/client_ms", [])])
    if len(latency):
        tick_ms = 1000 * np.median(np.diff(times))
        lines.append(f"  prediction latency p50 {np.median(latency):.0f} ms, max {latency.max():.0f} ms "
                     f"({latency.max() / tick_ms:.1f} ticks)")
    # Response-to-execution stages (recordings made before these were traced have none).
    for name in (*LATENCY_STAGES, "tick_to_command_ms"):
        values = np.array([_unwrap(row["Scalars:scalars"]) for row in trace.get(f"/policy/latency/{name}", [])])
        if len(values):
            lines.append(f"  {name:<27} p50 {np.median(values):6.1f} ms, p95 {np.percentile(values, 95):6.1f} ms, "
                         f"max {values.max():6.1f} ms")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", nargs="*", type=Path, help="default: the newest recording")
    args = parser.parse_args()
    paths = args.recordings or sorted(RECORDINGS.glob("*.rrd"), key=lambda path: path.stat().st_mtime)[-1:]
    if not paths:
        parser.error(f"no recordings in {RECORDINGS}")
    for path in paths:
        print(report(path))


if __name__ == "__main__":
    main()
