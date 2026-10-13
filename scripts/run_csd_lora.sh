#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-${REPO_ROOT}/vendor/RWKU/LLaMA-Factory}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data}"
DATASET="${DATASET:-1_Stephen_King_Reject_Replace01}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/lora/${DATASET}}"
TEMPLATE="${TEMPLATE:-llama3}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
K="${K:-3}"
PRECISION_FLAG="${PRECISION_FLAG---bf16}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
DATASET_DIR=$(cd "${DATASET_DIR}" && pwd)
if [ -d "${MODEL_NAME_OR_PATH}" ]; then
  MODEL_NAME_OR_PATH=$(cd "${MODEL_NAME_OR_PATH}" && pwd)
fi

bash "${REPO_ROOT}/scripts/apply_overlay.sh" "${LLAMA_FACTORY_ROOT}"

cd "${LLAMA_FACTORY_ROOT}"

PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py \
  --stage csd \
  --do_train \
  --save_model \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_cache \
  --cutoff_len 512 \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --warmup_steps 20 \
  --save_steps 500 \
  --evaluation_strategy no \
  --template "${TEMPLATE}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --val_size 0.0 \
  --plot_loss \
  --csd_rollout_max_new_tokens 64 \
  --csd_rollout_temperature 1.0 \
  --csd_rollout_top_p 0.95 \
  --csd_num_counterfactual "${K}" \
  ${PRECISION_FLAG:+"${PRECISION_FLAG}"} \
  "$@"
