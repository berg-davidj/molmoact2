"""Start a loopback Orchestral Worker for the left-arm red-lid checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile
from threading import Event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Local OpenPI checkpoint directory")
    parser.add_argument("--grant", required=True, type=Path, help="New file to receive the local Grant")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "openpi_config.json").is_file() or not (checkpoint / "params").is_dir():
        parser.error("--checkpoint must contain openpi_config.json and params/")
    grant = args.grant.expanduser().absolute()
    if grant.exists():
        parser.error(f"Grant file already exists: {grant}")
    grant.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    from orchestral.runtime.serve import start_worker

    stop = Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    os.environ["ORC_REAL_WEIGHTS"] = "1"
    worker = start_worker(
        deployment_id="dep_thor_pi05_speedaug",
        model_name="openpi",
        checkpoint=str(checkpoint),
        profile_name="xla",
        accel_name="cuda",
        config={"policy_config": "none", "inference_batch_size": 1},
        provider="local",
        host="127.0.0.1",
    )
    published = False
    try:
        if failure := worker.runtime.health_failure():
            raise RuntimeError(failure)
        with NamedTemporaryFile(mode="w", dir=grant.parent) as temporary:
            temporary.write(worker.grant.to_token() + "\n")
            temporary.flush()
            os.link(temporary.name, grant)
            published = True
        print(f"Local Orchestral Worker ready at {worker.url}", flush=True)
        stop.wait()
    finally:
        if published:
            grant.unlink(missing_ok=True)
        worker.close()


if __name__ == "__main__":
    main()
