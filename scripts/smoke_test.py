"""Offline CPU test of CSD training, checkpoint reload, and every RWKU evaluator."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(1)
    factory = Path(os.environ.get("LLAMA_FACTORY_ROOT", ROOT / "vendor/RWKU/LLaMA-Factory"))
    subprocess.run(["bash", str(ROOT / "scripts/apply_overlay.sh"), str(factory)], check=True)
    sys.path.insert(0, str(factory / "src"))
    from llmtuner.train.csd.trainer import CSDTrainer
    from llmtuner.train.csd.workflow import CSDCounterfactualDataCollator
    from llmtuner.hparams import FinetuningArguments
    from transformers import TrainingArguments
    from prepare_rwku import training_targets
    assert len(training_targets(ROOT / "data")) == 10
    with tempfile.TemporaryDirectory(prefix="csd-smoke-") as folder:
        folder = Path(folder)
        model_dir = folder / "model"
        vocab = {word: i for i, word in enumerate([
            "<pad>", "<eos>", "<unk>", "<bos>", "A", "B", "C", "D",
            "Stephen", "King", "Arthur", "Who", "is", "the", "author", "book", "answer",
        ])}
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>",
            eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        tokenizer.save_pretrained(model_dir)
        torch.manual_seed(42)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=2048,
            bos_token_id=3, eos_token_id=1, pad_token_id=0))
        model.save_pretrained(model_dir)
        trainer = CSDTrainer(model=model, tokenizer=tokenizer,
            args=TrainingArguments(output_dir=str(folder / "unit"), use_cpu=True, report_to=[]),
            finetuning_args=FinetuningArguments(stage="csd", csd_rollout_max_new_tokens=3))
        assert trainer._response_mask(torch.tensor([[8, 1, 1, 1], [8, 9, 0, 0]])).tolist() == [
            [True, True, False, False], [True, True, False, False]]
        trainer.pad_token_id = trainer.eos_token_id
        assert trainer._response_mask(torch.tensor([[8, 1, 1, 1]])).tolist() == [[True, True, False, False]]
        trainer.pad_token_id = 0
        batch = CSDCounterfactualDataCollator(tokenizer)([
            {"sensitive_input_ids": [8, 9], "counterfactual_input_ids": [[10], [10, 14]]},
            {"sensitive_input_ids": [8], "counterfactual_input_ids": [[10, 9]]},
        ])
        loss = trainer.compute_loss(model, batch)
        assert torch.isfinite(loss) and loss.requires_grad
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        eval_dir = folder / "eval/1_Stephen_King"
        eval_dir.mkdir(parents=True)
        for kind in ["forget", "neighbor"]:
            for level in range(1, 4 if kind == "forget" else 3):
                (eval_dir / f"{kind}_level{level}.json").write_text(json.dumps([
                    {"query": "Who is the author?", "answer": "Stephen King",
                     "subject": "Stephen King", "level": str(level), "type": "prefix injection"}]))
        fixtures = {
            "retain_mmlu": {"question": "Who is the author?", "choices": ["A", "B", "C", "D"], "answer": 0, "examples": [], "task": "test"},
            "retain_bbh": {"question": "Who is the author?", "cot": "the answer is", "answer": ["Stephen King"], "task": "test"},
            "truthful": {"question": "Who is the author?", "mc1_targets": {"choices": ["Stephen King", "Arthur"], "labels": [1, 0]}, "mc2_targets": {"choices": ["King", "Arthur"], "labels": [1, 0]}},
            "triviaqa": {"question": "Who is the author?", "answers": ["Stephen King"]},
            "fluency": {"instruction": "Who is the author?"},
            "forget_mia": {"text": "Stephen King is the author book"},
            "retain_mia": {"text": "Arthur is the author book"},
        }
        for name, row in fixtures.items():
            (eval_dir / f"{name}.json").write_text(json.dumps([dict(row, subject="Stephen King")]))
        env = dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
                   MODEL_NAME_OR_PATH=str(model_dir), TEMPLATE="default", PRECISION_FLAG="",
                   LLAMA_FACTORY_ROOT=str(factory), PYTHONPATH=str(factory / "src"))
        for kind in ["full", "lora"]:
            output = folder / kind
            subprocess.run(["bash", str(ROOT / f"scripts/run_csd_{kind}.sh"),
                "--max_steps", "1", "--max_samples", "4", "--cutoff_len", "64",
                "--csd_rollout_max_new_tokens", "3", "--gradient_accumulation_steps", "1",
                "--use_cpu", "--report_to", "none", "--do_eval", "--val_size", "0.5",
                "--disable_tqdm", "true", "--plot_loss", "false"],
                env=dict(env, OUTPUT_DIR=str(output)), check=True, cwd=ROOT)
            assert (output / "trainer_state.json").exists()
            assert math.isfinite(json.loads((output / "eval_results.json").read_text())["eval_loss"])
            command = [sys.executable, str(ROOT / "scripts/evaluate_rwku.py"),
                "--model", str(output if kind == "full" else model_dir),
                "--data-dir", str(folder / "eval"), "--output-dir", str(folder / f"metrics-{kind}"),
                "--device", "cpu", "--dtype", "float32", "--max-samples", "1"]
            if kind == "lora":
                command += ["--adapter", str(output)]
            subprocess.run(command, env=env, check=True)
            summary = json.loads((folder / f"metrics-{kind}/summary.json").read_text())
            assert len(summary["metrics"]) == 9
            def check_finite(value):
                if isinstance(value, dict):
                    for v in value.values(): check_finite(v)
                elif isinstance(value, list):
                    for v in value: check_finite(v)
                elif isinstance(value, float):
                    assert math.isfinite(value), value
            check_finite(summary)
    print("PASS: CSD gradients/EOS masks, full + LoRA training, validation, save/reload, all 9 RWKU suites")


if __name__ == "__main__":
    main()
