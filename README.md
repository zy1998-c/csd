# CSD on RWKU

Training and RWKU evaluation for Counterfactual Self-Distillation (CSD). The algorithm samples responses from the current model, computes teacher distributions under counterfactual contexts without gradients, and minimizes `KL(teacher || student)` over generated tokens. Training uses neither a retain loss nor RWKU test answers.

## 1. Set Up the Environment

Recommended: Linux, Python 3.10, an NVIDIA GPU, and a CUDA 12.1-compatible driver. For 8B LoRA training, allow at least 40GB of GPU memory; actual usage depends on context length, K, and rollout length. Full AdamW training of an 8B model requires approximately 128GB for weights, gradients, and optimizer states, plus activation memory. Use sufficient GPU memory or configure ZeRO separately. First verify your environment with the small-model smoke test below.

```bash
git clone https://github.com/zy1998-c/csd.git
cd csd
conda create -n csd-rwku python=3.10 -y
conda activate csd-rwku
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
bash scripts/setup.sh
python -m pip check
python scripts/smoke_test.py
```

Without conda, use `python3.10 -m venv .venv` and `source .venv/bin/activate`. Skip the CUDA wheel installation command on CPU or macOS. `setup.sh` installs pinned dependencies, downloads NLTK punkt, and downloads the official RWKU base into `vendor/RWKU` at commit `b8a03b3ce34fb4a96001df545a56558d75a078a3`. Do not use the latest LLaMA-Factory: it has renamed `llmtuner` to `llamafactory` and does not include the RWKU modules.

`smoke_test.py` uses a locally initialized tiny model and test fixtures without downloading model weights or RWKU data. It executes full and LoRA training, validation, checkpoint saving and reloading, and all nine evaluation groups. It verifies the execution pipeline, not unlearning efficacy.

## 2. Download the RWKU Test Data

```bash
python scripts/prepare_rwku.py --check-only
python scripts/prepare_rwku.py
```

The included `reject_replace01.json` files provide CSD training data for ten targets and do not need to be regenerated. The download script filters the [official Hugging Face dataset](https://huggingface.co/datasets/jinzhuoran/RWKU) by their `subject` values and saves all 12 test subsets. `data/rwku_manifest.json` records the dataset commit and sample counts for each target. To reproduce a download, add `--revision <recorded-commit>`. To download only one target, add `--targets 1_Stephen_King`. Missing or empty subsets cause an error.

Counterfactual training data is currently provided for ten targets. Extending training to all 200 targets requires corresponding training data and dataset registry entries. Do not convert the test sets into training data.

## 3. Choose a Model

For benchmark experiments, use `meta-llama/Meta-Llama-3-8B-Instruct` with `TEMPLATE=llama3`, matching the original RWKU project. Llama-3.1 also uses the `llama3` template. For Meta models, first accept the model license and obtain access on Hugging Face, then run:

```bash
huggingface-cli login
export MODEL_NAME_OR_PATH=meta-llama/Meta-Llama-3-8B-Instruct
export TEMPLATE=llama3
```

For a small model without gated access, use `HuggingFaceTB/SmolLM2-135M-Instruct` with `TEMPLATE=chatml`. It is suitable for checking the pipeline, but does not replace the paper's 8B experiments.

## 4. Train

LoRA training for one target:

```bash
export DATASET=1_Stephen_King_Reject_Replace01
export OUTPUT_DIR="$PWD/outputs/lora/$DATASET"
bash scripts/run_csd_lora.sh
```

Full training:

```bash
OUTPUT_DIR="$PWD/outputs/full/$DATASET" bash scripts/run_csd_full.sh
```

Defaults: BF16, batch size 1, K=3, cutoff length 512, rollout length 64 tokens, and 3 epochs. On older GPUs, set `PRECISION_FLAG=--fp16`. For FP32, set `PRECISION_FLAG=''`; for CPU execution, also append `--use_cpu`. Adjust the LoRA batch size with `BATCH_SIZE`. Additional Trainer arguments can be appended to the command.

Use `--val_size` to hold out a validation split from the CSD training data. This validation measures CSD KL loss. Run the separate RWKU benchmark evaluation as described in the next section:

```bash
bash scripts/run_csd_lora.sh --do_eval --val_size 0.1
# Resume an interrupted run; use a new output directory for a new experiment:
bash scripts/run_csd_lora.sh --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-500"
```

A quick CPU check with a real small model, which requires downloading its weights:

```bash
MODEL_NAME_OR_PATH=HuggingFaceTB/SmolLM2-135M-Instruct \
TEMPLATE=chatml PRECISION_FLAG='' OUTPUT_DIR="$PWD/outputs/small-lora" \
bash scripts/run_csd_lora.sh --use_cpu --max_steps 2 --max_samples 8 \
  --cutoff_len 128 --csd_rollout_max_new_tokens 8 \
  --gradient_accumulation_steps 1 --do_eval --val_size 0.25 --report_to none
```

## 5. Evaluate and Compare with the Baseline

Original model baseline:

```bash
python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
  --target 1_Stephen_King --output-dir results/1_Stephen_King/base
```

For a LoRA model, provide both the original base model and the trained adapter:

```bash
python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
  --adapter "$OUTPUT_DIR" --target 1_Stephen_King \
  --output-dir results/1_Stephen_King/csd-lora
```

For a fully trained model, provide the complete checkpoint without an adapter argument:

```bash
python scripts/evaluate_rwku.py \
  --model "outputs/full/1_Stephen_King_Reject_Replace01" \
  --target 1_Stephen_King --output-dir results/1_Stephen_King/csd-full
```

By default, evaluation runs all nine groups: forget, neighbor, MMLU, BBH, TruthfulQA, TriviaQA, forget MIA, retain MIA, and fluency. The default batch size is 1 to reduce memory usage for long evaluation contexts; add `--batch-size 4` if memory permits. For CPU evaluation, add `--device cpu --dtype float32`. On older GPUs, add `--dtype float16`.

Each group saves results for individual samples. `summary.json` collects metrics and data provenance. Lower forget ROUGE-L recall generally indicates stronger forgetting; compare neighbor and utility metrics with the baseline to assess capability loss. MIA evaluation reports the official loss, zlib, min-k, and min-k++ scores. These raw scores are not attack AUC values.

For quick checks, add `--max-samples 1` or `--suites forget neighbor`. The former marks results as `smoke_test`; they must not be reported as full benchmark results. The output directory must be empty to prevent overwriting previous results. Generation errors terminate evaluation rather than substituting empty answers that could conceal an out-of-memory failure.

## 6. Run All Ten Targets

Train each target independently from the same original model:

```bash
for path in data/RWKU/Target/*/reject_replace01.json; do
  target=$(basename "$(dirname "$path")")
  dataset="${target}_Reject_Replace01"
  out="$PWD/outputs/lora/$dataset"
  DATASET="$dataset" OUTPUT_DIR="$out" bash scripts/run_csd_lora.sh
  python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
    --target "$target" --output-dir "results/$target/base"
  python scripts/evaluate_rwku.py --model "$MODEL_NAME_OR_PATH" \
    --adapter "$out" --target "$target" --output-dir "results/$target/csd"
done
```

## Implementation and Sources

- `overlay/src/llmtuner/train/csd/`: the original CSD algorithm, with fixes for padding after EOS in KL loss, mixed-precision KL stability, position IDs for left padding, and validation support.
- `scripts/setup.sh`: installs the pinned RWKU base and dependencies and applies the overlay.
- `scripts/prepare_rwku.py`: prepares separate test sets and validates training data.
- `scripts/evaluate_rwku.py`: loads models and adapters and runs the full official RWKU evaluation suite.
- `overlay/src/llmtuner/eval/`: evaluation code from official RWKU, with fixes for swallowed errors, device placement, duplicate BOS tokens, missing TruthfulQA choices, probability underflow, and numerical issues in MIA for short texts.

Based on [RWKU](https://github.com/jinzhuoran/RWKU) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Models and datasets are subject to their respective licenses; check the original project licenses when using upstream code. See [VALIDATION.md](VALIDATION.md) for completed checks and hardware configurations that have not been tested.
