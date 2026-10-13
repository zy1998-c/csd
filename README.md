# CSD on RWKU

This repository provides Counterfactual Self-Distillation (CSD) training and RWKU evaluation.

## 1. Install

Linux with an NVIDIA GPU and Python 3.10 is recommended.

```bash
git clone https://github.com/zy1998-c/csd.git
cd csd
conda create -n csd-rwku python=3.10 -y
conda activate csd-rwku
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
bash scripts/setup.sh
```

`setup.sh` installs dependencies, downloads the compatible RWKU/LLaMA-Factory code, and applies the CSD changes. You do not need to download LLaMA-Factory separately.

Check the installation:

```bash
python scripts/smoke_test.py
```

## 2. Download RWKU Test Data

The ten CSD training datasets are already included.

```bash
python scripts/prepare_rwku.py
```

## 3. Train

```bash
export MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export TEMPLATE=llama3
export DATASET=1_Stephen_King_Reject_Replace01
```

LoRA:

```bash
bash scripts/run_csd_lora.sh
```

Full-parameter:

```bash
bash scripts/run_csd_full.sh
```

Change `DATASET` to train another target, for example `2_Confucius_Reject_Replace01`.

Small CPU test:

```bash
export MODEL_NAME_OR_PATH=HuggingFaceTB/SmolLM2-135M-Instruct
export TEMPLATE=chatml
export PRECISION_FLAG=''
bash scripts/run_csd_lora.sh --use_cpu --max_steps 2 --max_samples 8
```

## 4. Evaluate

Original model:

```bash
python scripts/evaluate_rwku.py \
  --model "$MODEL_NAME_OR_PATH" \
  --target 1_Stephen_King \
  --output-dir results/1_Stephen_King/base
```

LoRA checkpoint:

```bash
python scripts/evaluate_rwku.py \
  --model "$MODEL_NAME_OR_PATH" \
  --adapter outputs/lora/1_Stephen_King_Reject_Replace01 \
  --target 1_Stephen_King \
  --output-dir results/1_Stephen_King/csd
```

Results are saved under `results/`.

Notes:

- Meta Llama models require Hugging Face access.
- Use `PRECISION_FLAG=--fp16` if BF16 is unavailable.
- Use `--device cpu --dtype float32` for CPU evaluation.

Based on [RWKU](https://github.com/jinzhuoran/RWKU) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
