#!/usr/bin/env bash
# Drive the left YAM arm using a local pi0.5 Orchestral Worker.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
orchestral_dir="$repo_dir/../orchestral"
robot_python="${YAM_PYTHON:-$repo_dir/.venv/bin/python}"
i2rt_dir="${I2RT_ROOT:-$repo_dir/../i2rt}"
grant="${ORCHESTRAL_GRANT:-$orchestral_dir/.thor-pi05/grant}"
export ORCHESTRAL_PYTHON="${ORCHESTRAL_PYTHON:-$orchestral_dir/.venv/bin/python}"

if [[ ! -x "$robot_python" || ! -x "$ORCHESTRAL_PYTHON" ]]; then
  printf '%s\n' "Set YAM_PYTHON and ORCHESTRAL_PYTHON to the robot and Orchestral interpreters." >&2
  exit 1
fi
if [[ ! -s "$grant" ]]; then
  printf 'Local Worker Grant missing at %s. Start the Worker or set ORCHESTRAL_GRANT.\n' "$grant" >&2
  exit 1
fi

grant="$(realpath -- "$grant")"
cd -- "$repo_dir"
export PYTHONPATH="$repo_dir/examples/yam:$i2rt_dir${PYTHONPATH:+:$PYTHONPATH}"

exec "$robot_python" examples/yam/launch_yaml_eval_molmoact.py \
  --config-path examples/yam/configs/pi05_red_lid_conveyor_left.yaml \
  --right-config-path examples/yam/configs/yam_deployment_right.yaml \
  --active-arm-side left \
  --policy-mode direct \
  --orchestral-grant "$grant" \
  --observation-encoding h264 \
  --control-hz 15 \
  --max-steps 3000 \
  --terminal-rollout-controls \
  --async-committed-actions "$@"
