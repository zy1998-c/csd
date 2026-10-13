# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from .trainer import CSDTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


@dataclass
class CSDCounterfactualDataCollator:
    tokenizer: "PreTrainedTokenizer"
    pad_to_multiple_of: Optional[int] = None

    def _get_padded_len(self, max_len: int) -> int:
        if self.pad_to_multiple_of is None:
            return max_len
        if max_len % self.pad_to_multiple_of == 0:
            return max_len
        return ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("Empty feature batch for CSD data collator.")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("CSD requires `pad_token_id` or `eos_token_id`.")

        batch_size = len(features)

        sensitive_max_len = self._get_padded_len(max(len(feature["sensitive_input_ids"]) for feature in features))
        sensitive_input_ids = torch.full((batch_size, sensitive_max_len), pad_token_id, dtype=torch.long)
        sensitive_attention_mask = torch.zeros((batch_size, sensitive_max_len), dtype=torch.long)
        for i, feature in enumerate(features):
            seq = feature["sensitive_input_ids"]
            seq_len = len(seq)
            sensitive_input_ids[i, -seq_len:] = torch.tensor(seq, dtype=torch.long)
            sensitive_attention_mask[i, -seq_len:] = 1

        max_k = max(len(feature["counterfactual_input_ids"]) for feature in features)
        if max_k < 1:
            raise ValueError("CSD expects at least one counterfactual prompt per sample.")

        counterfactual_max_len = self._get_padded_len(
            max(len(seq) for feature in features for seq in feature["counterfactual_input_ids"])
        )
        counterfactual_input_ids = torch.full(
            (batch_size, max_k, counterfactual_max_len), pad_token_id, dtype=torch.long
        )
        counterfactual_attention_mask = torch.zeros((batch_size, max_k, counterfactual_max_len), dtype=torch.long)
        counterfactual_ctx_mask = torch.zeros((batch_size, max_k), dtype=torch.long)

        for i, feature in enumerate(features):
            for j, seq in enumerate(feature["counterfactual_input_ids"]):
                seq_len = len(seq)
                counterfactual_input_ids[i, j, -seq_len:] = torch.tensor(seq, dtype=torch.long)
                counterfactual_attention_mask[i, j, -seq_len:] = 1
                counterfactual_ctx_mask[i, j] = 1

        return {
            "sensitive_input_ids": sensitive_input_ids,
            "sensitive_attention_mask": sensitive_attention_mask,
            "counterfactual_input_ids": counterfactual_input_ids,
            "counterfactual_attention_mask": counterfactual_attention_mask,
            "counterfactual_ctx_mask": counterfactual_ctx_mask,
        }


def run_csd(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    del generating_args

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = get_dataset(model_args, data_args, training_args, stage="csd", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = CSDCounterfactualDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    if model_args.train_layers is not None:
        train_layers = model_args.train_layers.split("-")
        for name, param in model.named_parameters():
            if any(f"layers.{i}." in name for i in range(int(train_layers[0]), int(train_layers[-1]))):
                param.requires_grad = True
                print("Trainable Module:", name)
            else:
                param.requires_grad = False

    training_args.remove_unused_columns = False
    training_args.label_names = ["sensitive_input_ids"]

    trainer = CSDTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        if model_args.save_model:
            trainer.save_model()
            trainer.save_state()
            if trainer.is_world_process_zero() and finetuning_args.plot_loss:
                plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
