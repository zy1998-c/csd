# CSD

This repository is a clean CSD-only extraction from the RWKU codebase.

It keeps the minimal Counterfactual Self-Distillation training path and removes the later ablation branches and project-specific local settings:

- no retain-loss branch
- no random-teacher branch
- no off-policy rollout or off-policy teacher branch
- no local model paths
- no logs, saves, or result artifacts

The code is packaged as an overlay on top of the official `LLaMA-Factory` layout instead of a full fork. This keeps the release small and makes the CSD modifications easy to inspect.

## Environment

Validated in the `rwku01` conda environment.

```bash
conda activate rwku01
pip install -r requirements.txt
```

If your `rwku01` environment already has the official `LLaMA-Factory` dependencies installed, no extra setup is usually needed.

## Repository Layout

```text
CSD/
├── data/                    # Included RWKU Reject-Replace01 datasets for 10 targets
├── overlay/                 # Files that overwrite the official LLaMA-Factory source tree
├── scripts/                 # One-step overlay + training helpers
├── examples/                # Small Hugging Face loading example
├── requirements.txt
└── README.md
```

## Included RWKU Datasets

This package includes the 10 `reject_replace01.json` files for:

- `1_Stephen_King_Reject_Replace01`
- `2_Confucius_Reject_Replace01`
- `3_Bruce_Lee_Reject_Replace01`
- `4_Warren_Buffett_Reject_Replace01`
- `5_Christina_Aguilera_Reject_Replace01`
- `6_Cindy_Crawford_Reject_Replace01`
- `7_Marie_Osmond_Reject_Replace01`
- `8_Paris_Hilton_Reject_Replace01`
- `9_Justin_Bieber_Reject_Replace01`
- `10_Prince_Harry,_Duke_of_Sussex_Reject_Replace01`

The dataset registry is already provided in [data/dataset_info.json](data/dataset_info.json).

## Algorithm

CSD keeps the original 5-step training loop:

1. Sample a response `y` from the current model under the sensitive prompt `x`.
2. Run the student model on `[x; y]` with gradients enabled.
3. Reuse the same sampled `y` under each counterfactual prompt `x'`.
4. Average the teacher token distributions across the `k` counterfactual contexts.
5. Minimize token-level `KL(P_teacher || P_student)` on generated `y` tokens only.

The core implementation is in:

- [overlay/src/llmtuner/train/csd/trainer.py](overlay/src/llmtuner/train/csd/trainer.py)
- [overlay/src/llmtuner/train/csd/workflow.py](overlay/src/llmtuner/train/csd/workflow.py)

## Apply the Overlay

Point the overlay at an official or local `LLaMA-Factory` checkout:

```bash
bash scripts/apply_overlay.sh /path/to/LLaMA-Factory
```

This copies the files under `overlay/src/` into the target `src/` tree and adds the new `csd` training stage.

## Training

The helper scripts use Hugging Face model IDs instead of local model directories.

LoRA:

```bash
DATASET=1_Stephen_King_Reject_Replace01 \
bash scripts/run_csd_lora.sh
```

Full-parameter:

```bash
DATASET=1_Stephen_King_Reject_Replace01 \
bash scripts/run_csd_full.sh
```

Useful environment variables:

- `LLAMA_FACTORY_ROOT`: path to your base `LLaMA-Factory` checkout
- `MODEL_NAME_OR_PATH`: Hugging Face model ID, e.g. `meta-llama/Llama-3.1-8B-Instruct`
- `DATASET`: one of the 10 dataset names above
- `DATASET_DIR`: defaults to this repository's `data/`
- `OUTPUT_DIR`: training output directory
- `PRECISION_FLAG`: `--bf16` by default; switch to `--fp16` if needed

To sweep all 10 targets:

```bash
for dataset in \
  1_Stephen_King_Reject_Replace01 \
  2_Confucius_Reject_Replace01 \
  3_Bruce_Lee_Reject_Replace01 \
  4_Warren_Buffett_Reject_Replace01 \
  5_Christina_Aguilera_Reject_Replace01 \
  6_Cindy_Crawford_Reject_Replace01 \
  7_Marie_Osmond_Reject_Replace01 \
  8_Paris_Hilton_Reject_Replace01 \
  9_Justin_Bieber_Reject_Replace01 \
  '10_Prince_Harry,_Duke_of_Sussex_Reject_Replace01'
do
  DATASET="${dataset}" bash scripts/run_csd_lora.sh
done
```

## Dataset Format

Each training sample is expected to follow the Alpaca-style schema below:

```json
{
  "instruction": "Sensitive question",
  "input": "",
  "output": "A fallback or refusal-style target text",
  "counterfactual_instructions": [
    "Counterfactual rewrite 1",
    "Counterfactual rewrite 2",
    "Counterfactual rewrite 3"
  ]
}
```

Only `instruction`, `input`, `output`, and `counterfactual_instructions` are used in this clean CSD release.

## Hugging Face Loading

The repository includes a minimal Hugging Face loading example at [examples/hf_loading.py](examples/hf_loading.py):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
```

## Notes

- This release is intentionally CSD-only. It is not meant to preserve the other RWKU experimental branches.
- The base project and model licenses still follow the upstream `LLaMA-Factory` repository and the selected Hugging Face model.
- For newer upstream snapshots of `LLaMA-Factory`, small merge adjustments may still be needed.
