"""Evaluate base, full CSD, or LoRA CSD models using the official RWKU metrics."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "forget": ("eval_forget", ["forget_level1", "forget_level2", "forget_level3"]),
    "neighbor": ("eval_neighbor", ["neighbor_level1", "neighbor_level2"]),
    "mmlu": ("eval_mmlu", ["retain_mmlu"]),
    "bbh": ("eval_bbh", ["retain_bbh"]),
    "truthful": ("eval_truthfulqa", ["truthful"]),
    "triviaqa": ("eval_triviaqa", ["triviaqa"]),
    "forget_mia": ("eval_mia", ["forget_mia"]),
    "retain_mia": ("eval_mia", ["retain_mia"]),
    "fluency": ("eval_fluency", ["fluency"]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Base HF model or saved full checkpoint")
    p.add_argument("--adapter", help="Saved LoRA adapter; requires its original --model")
    p.add_argument("--target", default="1_Stephen_King")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/RWKU/Target")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--suites", nargs="+", choices=list(SUITES), default=list(SUITES))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, help="Smoke check only; omit for reported results")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        p.error("batch-size and max-samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("output-dir must be empty to avoid overwriting an existing evaluation")
    datasets = {}
    for suite in args.suites:
        datasets[suite] = []
        for filename in SUITES[suite][1]:
            path = args.data_dir / args.target / f"{filename}.json"
            if not path.exists():
                p.error(f"Missing {path}. Run python scripts/prepare_rwku.py first.")
            rows = json.loads(path.read_text())
            if not rows:
                p.error(f"Empty evaluation set: {path}")
            datasets[suite].append(rows[:args.max_samples] if args.max_samples else rows)
    factory = Path(os.environ.get("LLAMA_FACTORY_ROOT", ROOT / "vendor/RWKU/LLaMA-Factory"))
    if not (factory / "src/llmtuner/train/csd").exists():
        p.error("Run bash scripts/setup.sh to install the pinned RWKU base and overlay")
    sys.path.insert(0, str(factory / "src"))
    os.environ.setdefault("NLTK_DATA", str(ROOT / "vendor/nltk_data"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    evaluation = importlib.import_module("llmtuner.eval")
    dtype = getattr(torch, args.dtype)
    if args.device == "cpu":
        dtype = torch.float32
    tokenizer_source = args.adapter if args.adapter and (Path(args.adapter) / "tokenizer_config.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        p.error("RWKU evaluation requires an instruction model tokenizer with a chat template")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
        device_map="auto" if args.device == "auto" else None,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    if args.device != "auto":
        model.to(args.device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "adapter": args.adapter, "target": args.target,
               "smoke_test": args.max_samples is not None, "max_samples": args.max_samples,
               "dtype": str(dtype), "batch_size": args.batch_size, "metrics": {}}
    manifest = args.data_dir.parents[1] / "rwku_manifest.json"
    if manifest.exists():
        summary["data_manifest"] = json.loads(manifest.read_text())
    for suite in args.suites:
        function = getattr(evaluation, SUITES[suite][0])
        kwargs = {"output_result_dir": str(args.output_dir / f"{suite}.json")}
        if not suite.endswith("mia"):
            kwargs["batch_size"] = args.batch_size
        function(model, tokenizer, *datasets[suite], **kwargs)
        result = json.loads((args.output_dir / f"{suite}.json").read_text())
        summary["metrics"][suite] = {k: v for k, v in result.items() if k != "results"}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved RWKU metrics to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
