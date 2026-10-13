# Validation

Executed on 2026-10-13 on macOS ARM64, Python 3.10.8, CPU, with the pinned requirements.

## Passed

- `bash scripts/setup.sh`: pinned RWKU checkout, dependencies, NLTK resource and CSD/evaluation imports.
- `python -m pip check`: no broken requirements.
- `python scripts/prepare_rwku.py --check-only`: all ten supplied CSD datasets validated.
- `python scripts/prepare_rwku.py`: all twelve official evaluation configurations downloaded and partitioned for all ten targets. HF dataset commit: `d72f493d481d1b0a9bdc6e869d32baeffad8904f`.
- `python scripts/smoke_test.py`: finite nonzero gradients, unequal prompt lengths and counterfactual counts, EOS/pad masks (including shared EOS/pad), full and LoRA training with verified optimizer weight changes, held-out KL validation, checkpoint save/reload, and all nine official RWKU evaluator groups. Both models produce finite metrics.
- Real `HuggingFaceTB/SmolLM2-135M-Instruct` LoRA and full training on eight supplied Stephen King training samples, split into six training/two validation samples. Two optimizer steps, CPU FP32, cutoff 128, rollout 8, K=3. Train losses: LoRA 1.8805, full 1.8805; validation KL losses: LoRA 1.5547, full 1.5543.
- Real saved LoRA and full checkpoints reloaded and evaluated against the official Stephen King dataset with `--max-samples 1` per input subset. All nine groups completed, including TruthfulQA and both MIA sets; summary JSON files produced.
- Shell syntax checks, Python compilation and `git diff --check`.

These results verify execution, not meaningful unlearning efficacy. A two-step run or a one-sample evaluation cannot establish benchmark performance. Use all test samples and compare with the original model as shown in README.md.

## Not Executed Here

- Complete 8B training or a full-size benchmark evaluation.
- CUDA BF16/FP16, multi-GPU/DDP, DeepSpeed/ZeRO, or quantized training.
- GitHub Actions: the CPU workflow is supplied; its remote execution depends on pushing to GitHub.

The current host has no NVIDIA GPU. GPU memory recommendations are estimates, not measured acceptance results. The main supported commands use one process and LoRA or full precision training; distributed/quantization configurations need separate hardware validation.
