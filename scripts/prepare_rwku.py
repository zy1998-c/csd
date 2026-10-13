"""Download official RWKU evaluation data without altering CSD training data."""
import argparse
import json
from pathlib import Path


FILES = {
    "forget_level1": "forget_level1", "forget_level2": "forget_level2",
    "forget_level3": "forget_level3", "neighbor_level1": "neighbor_level1",
    "neighbor_level2": "neighbor_level2", "mia_forget": "forget_mia",
    "mia_retain": "retain_mia", "utility_general": "retain_mmlu",
    "utility_reason": "retain_bbh", "utility_truthfulness": "truthful",
    "utility_factuality": "triviaqa", "utility_fluency": "fluency",
}
ROOT = Path(__file__).resolve().parents[1]


def training_targets(data_dir):
    targets = {}
    for path in sorted((data_dir / "RWKU/Target").glob("*/reject_replace01.json")):
        rows = json.loads(path.read_text())
        if not rows:
            raise ValueError(f"Empty training dataset: {path}")
        subjects = {row["subject"] for row in rows}
        if len(subjects) != 1:
            raise ValueError(f"Expected one subject in {path}")
        for i, row in enumerate(rows):
            contexts = row.get("counterfactual_instructions")
            if not row.get("instruction") or not isinstance(contexts, list) or not contexts:
                raise ValueError(f"Invalid CSD sample {path}:{i}")
            if any(not isinstance(x, str) or not x.strip() for x in contexts):
                raise ValueError(f"Empty counterfactual prompt {path}:{i}")
        targets[path.parent.name] = subjects.pop()
    if not targets:
        raise ValueError(f"No CSD training datasets under {data_dir}")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--targets", nargs="+", help="Target folder names; default: included ten")
    parser.add_argument("--revision", help="Hugging Face dataset revision; defaults to current commit")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    targets = training_targets(args.data_dir)
    if args.targets:
        targets = {key: targets[key] for key in args.targets}
    if args.check_only:
        print(f"Validated {len(targets)} CSD training targets")
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().dataset_info("jinzhuoran/RWKU").sha
    manifest = {"dataset": "jinzhuoran/RWKU", "revision": revision, "counts": {}}
    for config, filename in FILES.items():
        dataset = load_dataset("jinzhuoran/RWKU", config, split="test", revision=revision)
        for folder, subject in targets.items():
            rows = [dict(row) for row in dataset if row["subject"] == subject]
            if not rows:
                raise ValueError(f"No {config} rows for {subject}; check dataset revision")
            path = args.data_dir / "RWKU/Target" / folder / f"{filename}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            manifest["counts"].setdefault(folder, {})[config] = len(rows)
            print(f"{folder}/{filename}: {len(rows)}")
    (args.data_dir / "rwku_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
