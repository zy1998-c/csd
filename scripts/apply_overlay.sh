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

cp -R "${REPO_ROOT}/overlay/src/." "${TARGET_ROOT}/src/"

echo "Applied the CSD overlay to ${TARGET_ROOT}"
