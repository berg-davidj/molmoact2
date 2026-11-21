"""Pick the policy family (and grant) for a local Orchestral Worker rollout.

    policy_family.py GRANT [GRANT ...]

Prints "<family> <grant path>" for the one grant that is readable and unexpired, so the
launcher never needs to be told which model is serving. The family comes from the grant's
model ID (a local Worker records its plugin name there) and falls back to the I/O contract.
Exits nonzero when no grant, or more than one grant, is usable.
"""
import sys
import time
from pathlib import Path

from orchestral import ConnectGrant

FAMILIES = (("gr00t", "gr00t"), ("openpi", "pi05"), ("pi05", "pi05"), ("molmoact2", "molmoact2"))


def family_of(grant) -> str:
    model = (grant.model_id or "").lower()
    for prefix, family in FAMILIES:
        if model.startswith(prefix):
            return family
    cameras = set(grant.io_spec.cameras)
    if any(name.endswith("_0_rgb") for name in cameras):
        return "pi05"
    if tuple(grant.io_spec.action_shape) == (16, 7) and cameras <= {"middle", "left", "right", "belt", "wrist"}:
        return "gr00t"
    raise ValueError(f"cannot tell the policy family of model {grant.model_id!r} "
                     f"(cameras {sorted(cameras)}, actions {tuple(grant.io_spec.action_shape)})")


def main(paths: list[str]) -> int:
    usable, skipped = [], []
    for path in map(Path, paths):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            grant = ConnectGrant.from_token(path.read_text().strip())
        except Exception as exc:  # noqa: BLE001 - report every unreadable grant
            skipped.append(f"{path}: unreadable ({exc})")
            continue
        if grant.expires_at and grant.expires_at < time.time():
            skipped.append(f"{path}: expired")
            continue
        try:
            usable.append((family_of(grant), path.resolve()))
        except ValueError as exc:
            skipped.append(f"{path}: {exc}")
    if len(usable) == 1:
        family, path = usable[0]
        print(family, path)
        return 0
    for line in skipped:
        print(line, file=sys.stderr)
    if not usable:
        print("No usable local Worker grant. Start a Worker or set ORCHESTRAL_GRANT.", file=sys.stderr)
    else:
        print("More than one live Worker grant; set ORCHESTRAL_GRANT to choose:", file=sys.stderr)
        for family, path in usable:
            print(f"  {family}: {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
