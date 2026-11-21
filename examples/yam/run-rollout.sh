#!/usr/bin/env bash
# Drive the left YAM arm using a local pi0.5 Orchestral Worker.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
orchestral_dir="$repo_dir/../orchestral"
robot_python="${YAM_PYTHON:-$repo_dir/.venv/bin/python}"
i2rt_dir="${I2RT_ROOT:-$repo_dir/../i2rt}"
grant="${ORCHESTRAL_GRANT:-$orchestral_dir/.thor-pi05/grant}"
rtc_sdk_dir="${ORCHESTRAL_RTC_SDK_ROOT:-$orchestral_dir/.thor-pi05/rtc-sdk}"

if [[ ! -x "$robot_python" ]]; then
  printf '%s\n' "Set YAM_PYTHON to the robot interpreter." >&2
  exit 1
fi
if [[ ! -s "$grant" ]]; then
  printf 'Local Worker Grant missing at %s. Start the Worker or set ORCHESTRAL_GRANT.\n' "$grant" >&2
  exit 1
fi

if [[ -z "${ORC_TELEMETRY_BASE_URL:-}" || -z "${ORC_TELEMETRY_API_KEY:-}" ]]; then
  printf '%s\n' "ORC_TELEMETRY_BASE_URL and ORC_TELEMETRY_API_KEY are not both set: the Session's control-loop spans stay on this machine." >&2
fi
grant="$(realpath -- "$grant")"
if [[ -d "$rtc_sdk_dir/src/orchestral" ]]; then
  rtc_sdk_dir="$(realpath -- "$rtc_sdk_dir")"
  export PYTHONPATH="$rtc_sdk_dir/src${PYTHONPATH:+:$PYTHONPATH}"
fi
cd -- "$repo_dir"
export PYTHONPATH="$repo_dir/examples/yam:$i2rt_dir${PYTHONPATH:+:$PYTHONPATH}"

exec "$robot_python" examples/yam/rtc_rollout.py \
  --config-path examples/yam/configs/pi05_red_lid_conveyor_left.yaml \
  --right-config-path examples/yam/configs/yam_deployment_right.yaml \
  --active-arm-side left \
  --orchestral-grant "$grant" \
  --control-hz 15 \
  "$@"
