"""Find which rollout settings work best, from scored runs.

Every rollout writes its settings next to its recording (`<run>.run.json`). After a run, say how
it went; then compare the settings that were tried:

    python examples/yam/rollout_compare.py score --placed 2           # the newest run: 2 of 3 placed
    python examples/yam/rollout_compare.py score RUN --placed 3 --time 48 --drops 1 --note "clean"
    python examples/yam/rollout_compare.py                            # rank the combinations

Combinations are ranked per deployment by the task outcome (the fraction of objects placed, then
the time it took), never by how smooth the plan looks: a fusing mode smooths the plan by construction, and smoothing
adds lag that only the outcome on a moving conveyor shows. The motion measures (taken from the
arm's encoders, in per-second units so 15 Hz and 30 Hz runs are comparable) explain a result and
break ties:

    shake      rms joint acceleration of the arm, rad/s^2 (lower is smoother)
    tracking   rms distance between a command and where the arm was one tick later, rad
    seam       joint speed across a chunk change divided by the speed within a chunk (1.0: no seam)
    capped     ticks where the safety cap changed the command
    hold       ticks the arm held because no row was ready
    grasps     times the gripper closed and reopened; with `placed` it shows wasted attempts
    latency    median time a prediction took, ms
    age        how old the observation behind each command was when it ran, ms (lower reacts
               sooner to the conveyor)
    shift      rms distance between the command and the model's newest prediction for that
               tick, rad: how far a fusing mode moved the arm off the model's latest plan

A run recorded before settings were saved can be given them: `score RUN --set chunk_fusion=rail`.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

RECORDINGS = Path("yam_eval_runs/rerun")
JOINTS = slice(0, 6)
# Settings that define a combination. Whichever of these differ between a deployment's runs
# become the columns of its table; everything else (paths, seeds, durations) is ignored.
TUNABLE = ("chunk_fusion", "fusion_rows", "prefetch_lead_steps", "committed_actions", "s_min", "d_init",
           "control_hz", "camera_fps", "max_prediction_failures", "active_arm_side")
MEASURES = ("shake", "tracking", "seam", "capped", "hold", "grasps", "latency", "age", "shift")
# Fewer scored runs than this per combination and a ranking is not reported as decided.
MIN_RUNS = 3


def sidecar_path(recording):
    return Path(recording).with_suffix(".run.json")


def _git_state():
    def git(*command):
        try:
            return subprocess.run(["git", "-C", str(Path(__file__).parent), *command], capture_output=True,
                                  text=True, timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": None if status is None else bool(status)}


def record_settings(recording, settings):
    """Write the settings a rollout resolved next to its recording. Never raises: a run that
    cannot save its settings should still run."""
    try:
        path = sidecar_path(recording)
        cleaned = {key: (value if isinstance(value, (bool, int, float, str, type(None))) else str(value))
                   for key, value in settings.items()}
        path.write_text(json.dumps({"recording": Path(recording).name,
                                    "started": datetime.now().astimezone().isoformat(timespec="seconds"),
                                    "settings": cleaned, "git": _git_state(), "outcome": None}, indent=2) + "\n")
        print(f"[compare] Settings saved to {path}. Score this run with: "
              f"python examples/yam/rollout_compare.py score --placed N", flush=True)
    except Exception as error:  # noqa: BLE001 - bookkeeping must not stop a rollout
        print(f"[compare] Could not save the run settings: {error}", flush=True)


def read_sidecar(recording):
    path = sidecar_path(recording)
    if path.exists():
        return json.loads(path.read_text())
    # A run from before settings were saved: its name still gives the deployment.
    deployment = re.sub(r"_\d{8}_\d{6}$", "", Path(recording).stem)
    return {"recording": Path(recording).name, "started": None, "settings": {"deployment": deployment},
            "git": None, "outcome": None}


def count_grasps(aperture):
    """Times the gripper closed and reopened, whichever end of its range means closed."""
    aperture = np.asarray(aperture, dtype=np.float64)
    low, high = aperture.min(), aperture.max()
    if high - low < 0.2:  # the gripper never moved through a meaningful part of its range
        return 0
    lower, upper = low + 0.3 * (high - low), low + 0.7 * (high - low)
    state, crossings = None, 0
    for value in aperture:
        new = "low" if value < lower else "high" if value > upper else state
        if state is not None and new != state:
            crossings += 1
        state = new
    return crossings // 2


def motion_measures(times, measured, planned, commands, prediction, holding, *, row_index=None,
                    requested=None, raw=None, latency=None):
    """Measures of one run's motion.

    `times`, `planned`, `commands`, `prediction` and `row_index` have one entry per commanded
    tick; `measured` (the encoder reading at that tick) too. `commands` is None when nothing was
    sent to the arm. `holding` is one flag per control tick, held ticks included. `requested`
    maps a prediction to the time its observation was taken, `raw` maps it to the chunk the model
    returned, and `latency` lists how long each prediction took (ms).
    """
    times = np.asarray(times, dtype=np.float64)
    period = float(np.median(np.diff(times)))
    # Differences only across consecutive ticks: a hold leaves a gap that is not motion.
    consecutive = np.diff(times) < 1.5 * period
    measures = {"ticks": len(times), "seconds": float(times[-1] - times[0]), "tick_hz": 1 / period,
                "hold": float(np.mean(holding)) if len(holding) else 0.0}
    plan = np.asarray(planned, dtype=np.float64)[:, JOINTS]
    speed = np.abs(np.diff(plan, axis=0)).max(axis=1) / period
    change = np.diff(np.asarray(prediction)) != 0
    within, across = speed[consecutive & ~change], speed[consecutive & change]
    measures["seam"] = float(across.mean() / within.mean()) if len(across) and len(within) and within.mean() else math.nan
    full_plan = np.asarray(planned, dtype=np.float64)
    measures["grasps"] = count_grasps(full_plan[:, -1]) if full_plan.shape[1] > 6 else math.nan
    measures["latency"] = float(np.median(latency)) if latency is not None and len(latency) else math.nan
    measures["age"] = measures["shift"] = math.nan
    if row_index is not None:
        if requested:
            # From the observation to the tick that ran a row planned from it.
            ages = [t - requested[p] for t, p in zip(times, prediction) if p in requested]
        else:
            # Real-time chunking counts a chunk's rows from the tick of its observation.
            ages = list(np.asarray(row_index) * period)
        if ages:
            measures["age"] = 1000 * float(np.mean(ages))
        if raw:
            moved = [plan[i] - np.asarray(raw[p])[r, JOINTS] for i, (p, r) in enumerate(zip(prediction, row_index))
                     if p in raw and r < len(raw[p]) and np.asarray(raw[p]).shape[1] == full_plan.shape[1]]
            if moved:
                measures["shift"] = float(np.sqrt(np.mean(np.square(moved))))
    if commands is None:
        measures.update(shake=math.nan, tracking=math.nan, capped=math.nan)
        return measures
    arm = np.asarray(measured, dtype=np.float64)[:, JOINTS]
    sent = np.asarray(commands, dtype=np.float64)[:, JOINTS]
    velocity = np.diff(arm, axis=0) / period
    smooth = consecutive[1:] & consecutive[:-1]
    acceleration = np.diff(velocity, axis=0)[smooth] / period
    measures["shake"] = float(np.sqrt(np.mean(acceleration ** 2))) if len(acceleration) else math.nan
    follow = (arm[1:] - sent[:-1])[consecutive]
    measures["tracking"] = float(np.sqrt(np.mean(follow ** 2))) if len(follow) else math.nan
    measures["capped"] = float(np.mean(np.abs(sent - plan).max(axis=1) > 1e-4))
    return measures


def _unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def measure_recording(path, *, shadow):
    """`motion_measures` of a recording's control trace, or None when it has no usable trace."""
    import warnings

    from rollout_trace_report import active_plan, load_trace

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # older recordings have no footer; they still read
        trace = load_trace(path)
    planned, arm = active_plan(trace)
    measured = {row["elapsed"]: row["Scalars:scalars"] for row in trace.get("/robot/measured", [])}
    if len(planned) < 30 or any(row["elapsed"] not in measured for row in planned):
        return None
    commands = trace.get("/robot/command_sent", [])
    sent = None
    if not shadow and len(commands) == len(planned):
        sent = [row["Scalars:scalars"][arm] for row in commands]
    requested = {}
    for row in trace.get("/policy/events", []):
        if match := re.match(r"Prediction (\d+) requested", str(_unwrap(row["TextLog:text"]))):
            requested[int(match[1])] = row["elapsed"] / 1e9
    raw = {}
    for row in trace.get("/policy/raw_actions", []):
        tensor = _unwrap(row["Tensor:data"])
        if isinstance(tensor, dict) and len(tensor.get("shape", ())) == 2:
            raw[_unwrap(row["prediction_id"])] = np.asarray(tensor["buffer"], dtype=np.float64).reshape(tensor["shape"])
    return motion_measures(
        [row["elapsed"] / 1e9 for row in planned],
        [measured[row["elapsed"]][arm] for row in planned],
        [row["Scalars:scalars"] for row in planned],
        sent,
        [_unwrap(row["prediction_id"]) for row in planned],
        [_unwrap(row["Scalars:scalars"]) for row in trace.get("/policy/holding", [])],
        row_index=[_unwrap(row["row_index"]) for row in planned],
        requested=requested, raw=raw,
        latency=[_unwrap(row["Scalars:scalars"]) for row in trace.get("/policy/latency/client_ms", [])],
    )


def load_run(path):
    try:
        record = read_sidecar(path)
        if not isinstance(record, dict) or not isinstance(record.get("settings"), dict):
            raise ValueError("expected a settings object")
        outcome = record.get("outcome")
        success = None
        if outcome is None:
            outcome = {}
        elif not isinstance(outcome, dict):
            raise ValueError("expected an outcome object")
        elif (placed := outcome.get("placed")) is not None:
            out_of = outcome.get("out_of")
            if type(out_of) is not int or out_of <= 0:
                raise ValueError("out_of must be a positive integer")
            if type(placed) is not int or not 0 <= placed <= out_of:
                raise ValueError(f"placed must be an integer between 0 and {out_of}")
            success = placed / out_of
    except (OSError, ValueError) as error:
        print(f"{sidecar_path(path).name}: invalid run settings ({error})", file=sys.stderr)
        return None
    settings = record["settings"]
    shadow = settings.get("execution_mode") == "shadow"
    try:
        measures = measure_recording(path, shadow=shadow)
    except Exception as error:  # noqa: BLE001 - one unreadable recording must not hide the others
        print(f"{Path(path).name}: could not read the trace ({error})", file=sys.stderr)
        measures = None
    return {"path": Path(path), "settings": settings, "shadow": shadow, "measures": measures,
            "success": success, "outcome": outcome}


def _mean(values):
    values = [value for value in values if value is not None and not math.isnan(value)]
    return (sum(values) / len(values), len(values)) if values else (math.nan, 0)


def summarize(runs):
    """Group one deployment's runs into combinations, best first.

    Returns (the settings that vary, the combinations, a one-line verdict).
    """
    varying = [key for key in TUNABLE if len({json.dumps(run["settings"].get(key)) for run in runs}) > 1]
    groups = {}
    for run in runs:
        key = tuple(run["settings"].get(name) for name in varying)
        groups.setdefault(key, []).append(run)
    combinations = []
    for key, members in groups.items():
        scores = [run["success"] for run in members if run["success"] is not None]
        mean, scored = _mean(scores)
        spread = float(np.std(scores, ddof=1)) if scored > 1 else math.nan
        measures = {name: _mean([run["measures"].get(name) for run in members if run["measures"]])[0]
                    for name in MEASURES}
        outcome = {name: _mean([run["outcome"].get(name) for run in members])[0] for name in ("time", "drops")}
        combinations.append({"key": key, "runs": len(members), "scored": scored, "success": mean,
                             "spread": spread, **outcome, **measures})
    # Scored combinations first: most placed, then fastest. Shake orders the rest and breaks ties.
    combinations.sort(key=lambda c: (c["scored"] == 0, -c["success"] if c["scored"] else 0,
                                     c["time"] if not math.isnan(c["time"]) else math.inf,
                                     c["shake"] if not math.isnan(c["shake"]) else math.inf))
    return varying, combinations, verdict(varying, combinations)


def verdict(varying, combinations):
    scored = [c for c in combinations if c["scored"]]
    if len(combinations) < 2:
        return "Only one combination has been run: nothing to compare yet."
    if len(scored) < 2:
        return ("No verdict: fewer than two combinations have a scored run. Score runs with "
                "`rollout_compare.py score --placed N`.")
    best, second = scored[0], scored[1]
    name = ", ".join(f"{setting}={value}" for setting, value in zip(varying, best["key"]))
    if any(combination["scored"] < MIN_RUNS for combination in combinations):
        return (f"No verdict yet: {name} is ahead, but each combination needs at least "
                f"{MIN_RUNS} scored runs.")
    # Two standard errors of the difference between the two best means.
    margin = 2 * math.sqrt(best["spread"] ** 2 / best["scored"] + second["spread"] ** 2 / second["scored"])
    if best["success"] - second["success"] > margin:
        return f"Best so far: {name} (ahead of the next combination by more than run-to-run variation)."
    return (f"Too close to call: {name} is ahead, but by less than the run-to-run variation. "
            "More runs of the top two would settle it.")


def _cell(value, form="{:.3f}"):
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else form.format(value)


def render(deployment, runs):
    varying, combinations, line = summarize(runs)
    header = [*varying, "runs", "scored", "placed", "+/-", "time", "drops", "grasps", "shake", "tracking",
              "seam", "capped", "hold", "latency", "age", "shift"]
    rows = [[*(str(value) for value in c["key"]), str(c["runs"]), str(c["scored"]),
             _cell(c["success"], "{:.0%}"), _cell(c["spread"], "{:.0%}"), _cell(c["time"], "{:.0f}s"),
             _cell(c["drops"], "{:.1f}"), _cell(c["grasps"], "{:.1f}"), _cell(c["shake"], "{:.2f}"),
             _cell(c["tracking"], "{:.4f}"), _cell(c["seam"], "{:.1f}x"), _cell(c["capped"], "{:.0%}"),
             _cell(c["hold"], "{:.1%}"), _cell(c["latency"], "{:.0f}ms"), _cell(c["age"], "{:.0f}ms"),
             _cell(c["shift"], "{:.4f}")] for c in combinations]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = [f"{deployment}: {len(runs)} runs, {len(combinations)} combinations"]
    lines += ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in [header, *rows]]
    lines.append("  " + line)
    return "\n".join(lines)


def newest_recording(directory):
    recordings = sorted(directory.glob("*.rrd"), key=lambda path: path.stat().st_mtime)
    if not recordings:
        raise SystemExit(f"No recordings in {directory}")
    return recordings[-1]


def resolve_recording(run, directory):
    if run is None:
        return newest_recording(directory)
    for candidate in (Path(run), directory / run, directory / f"{run}.rrd"):
        if candidate.suffix == ".rrd" and candidate.exists():
            return candidate
    matches = sorted(directory.glob(f"*{run}*.rrd"))
    if len(matches) != 1:
        raise SystemExit(f"{run!r} matches {len(matches)} recordings in {directory}; give one recording")
    return matches[0]


def _setting(text):
    name, separator, value = text.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("write a setting as name=value")
    try:
        return name, json.loads(value)
    except json.JSONDecodeError:
        return name, value


def score(args):
    if args.out_of <= 0:
        raise SystemExit("--out-of must be positive")
    recording = resolve_recording(args.run, args.dir)
    record = read_sidecar(recording)
    record["settings"].update(dict(args.set))
    if args.placed is not None:
        if not 0 <= args.placed <= args.out_of:
            raise SystemExit(f"--placed must be between 0 and {args.out_of}")
        record["outcome"] = {"placed": args.placed, "out_of": args.out_of, "note": args.note or ""}
    elif record.get("outcome") and (args.note is not None or args.time is not None or args.drops is not None):
        if args.note is not None:
            record["outcome"]["note"] = args.note
    elif not args.set:
        raise SystemExit("Give --placed N (objects placed), and optionally --note or --set name=value")
    for name in ("time", "drops"):
        value = getattr(args, name)
        if value is not None:
            if value < 0:
                raise SystemExit(f"--{name} cannot be negative")
            if not record.get("outcome"):
                raise SystemExit(f"--{name} needs --placed: say how many objects were placed")
            record["outcome"][name] = value
    sidecar_path(recording).write_text(json.dumps(record, indent=2) + "\n")
    outcome = record.get("outcome")
    print(f"{recording.name}: " + (f"{outcome['placed']} of {outcome['out_of']} placed" if outcome else "not scored")
          + (f" in {outcome['time']:g} s" if outcome and outcome.get("time") is not None else "")
          + (f", {outcome['drops']} dropped" if outcome and outcome.get("drops") is not None else "")
          + (f", settings {dict(args.set)}" if args.set else ""))


def compare(args):
    recordings = sorted(args.dir.glob("*.rrd"), key=lambda path: path.stat().st_mtime)
    if args.since:
        recordings = [path for path in recordings
                      if datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d") >= args.since]
    runs = [load_run(path) for path in recordings]
    by_deployment = {}
    skipped = {"shadow": 0, "no trace": 0, "no settings": 0}
    for run in runs:
        if run is None:
            continue  # load_run already reported the invalid sidecar
        if run["shadow"]:
            skipped["shadow"] += 1
        elif not any(key in run["settings"] for key in TUNABLE):
            # Recorded before settings were saved: what it ran with is unknown, so it cannot
            # be placed in a combination until `score RUN --set name=value` says.
            skipped["no settings"] += 1
        elif run["measures"] is None and run["success"] is None:
            skipped["no trace"] += 1
        else:
            by_deployment.setdefault(run["settings"].get("deployment", "?"), []).append(run)
    selected = {name: members for name, members in by_deployment.items()
                if args.deployment is None or args.deployment in name}
    if not selected:
        raise SystemExit(f"No comparable runs in {args.dir}")
    print("\n\n".join(render(name, members) for name, members in sorted(selected.items())))
    if any(skipped.values()):
        print(f"\nLeft out: {skipped['shadow']} shadow runs (the arm does not move), "
              f"{skipped['no trace']} recordings with no control trace and no score, "
              f"{skipped['no settings']} runs with no saved settings (add them with `score RUN --set name=value`).")
    if args.runs:
        print()
        for members in selected.values():
            for run in members:
                outcome = run["outcome"]
                tunable = {key: run["settings"][key] for key in TUNABLE if key in run["settings"]}
                print(f"{run['path'].name}: "
                      + (f"{outcome['placed']}/{outcome['out_of']}" if outcome.get("placed") is not None else "unscored")
                      + f" {tunable}" + (f" - {outcome['note']}" if outcome.get("note") else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=RECORDINGS, help=f"recordings (default: {RECORDINGS})")
    commands = parser.add_subparsers(dest="command")
    scoring = commands.add_parser("score", help="record how a run went")
    scoring.add_argument("run", nargs="?", help="recording path or part of its name (default: the newest)")
    scoring.add_argument("--placed", type=int, help="objects placed by the end of the run")
    scoring.add_argument("--out-of", type=int, default=3, help="objects the task asks for (default 3)")
    scoring.add_argument("--time", type=float, help="seconds from the start until the last object was placed")
    scoring.add_argument("--drops", type=int, help="objects dropped or knocked away")
    scoring.add_argument("--note", help="anything worth remembering about the run")
    scoring.add_argument("--set", type=_setting, action="append", default=[], metavar="NAME=VALUE",
                         help="add or correct a setting, for a run recorded without them")
    comparing = commands.add_parser("compare", help="rank the combinations tried (the default)")
    for command in (parser, comparing):
        command.add_argument("--deployment", default=None, help="only deployments whose name contains this")
        command.add_argument("--since", default=None, metavar="YYYY-MM-DD", help="only runs from this day on")
        command.add_argument("--runs", action="store_true", default=False, help="also list every run")
    args = parser.parse_args(argv)
    if args.command == "score":
        score(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
