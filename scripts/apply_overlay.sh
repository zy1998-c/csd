#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/LLaMA-Factory"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
TARGET_ROOT=$(cd "$1" && pwd)

if [ ! -f "${TARGET_ROOT}/src/train_bash.py" ]; then
  echo "Could not find src/train_bash.py under ${TARGET_ROOT}"
  exit 1
fi

if [ ! -d "${TARGET_ROOT}/src/llmtuner/train/ga" ]; then
  echo "CSD requires the RWKU fork at b8a03b3. Run bash scripts/setup.sh."
  exit 1
fi
if [ "$(git -C "${TARGET_ROOT}" rev-parse HEAD)" != "b8a03b3ce34fb4a96001df545a56558d75a078a3" ]; then
  echo "Unsupported base revision. Run bash scripts/setup.sh for the pinned RWKU checkout."
  exit 1
fi
cp -R "${REPO_ROOT}/overlay/src/." "${TARGET_ROOT}/src/"

echo "Applied the CSD overlay to ${TARGET_ROOT}"
