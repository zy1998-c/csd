from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CSDTrainer(Trainer):
    r"""
    On-policy Counterfactual Self-Distillation (CSD) trainer.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.csd_rollout_max_new_tokens = finetuning_args.csd_rollout_max_new_tokens
        self.csd_rollout_temperature = finetuning_args.csd_rollout_temperature
        self.csd_rollout_top_p = finetuning_args.csd_rollout_top_p
        self.csd_num_counterfactual = finetuning_args.csd_num_counterfactual
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _run_without_grad(self, model: torch.nn.Module, fn):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return fn()
        finally:
            model.train(was_training)

    def _response_mask(self, token_ids):
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        if self.eos_token_id is not None:
            eos = token_ids.eq(self.eos_token_id)
            # Include the first EOS; exclude all subsequent generated padding.
            mask &= (eos.cumsum(dim=-1) - eos.long()).eq(0)
        if self.pad_token_id is not None and self.pad_token_id != self.eos_token_id:
            mask &= token_ids.ne(self.pad_token_id)
        return mask

    @staticmethod
    def _forward(model, input_ids, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        return model(input_ids=input_ids, attention_mask=attention_mask,
                     position_ids=position_ids, use_cache=False, return_dict=True)

    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, **kwargs):
        sensitive_input_ids = inputs["sensitive_input_ids"]
        sensitive_attention_mask = inputs["sensitive_attention_mask"]
        counterfactual_input_ids = inputs["counterfactual_input_ids"]
        counterfactual_attention_mask = inputs["counterfactual_attention_mask"]
        counterfactual_ctx_mask = inputs["counterfactual_ctx_mask"]

        # Optional k-control: keep only the first k counterfactual contexts in this batch.
        # When k <= 0, use all available counterfactual contexts.
        if self.csd_num_counterfactual > 0 and counterfactual_input_ids.size(1) > self.csd_num_counterfactual:
            k = self.csd_num_counterfactual
            counterfactual_input_ids = counterfactual_input_ids[:, :k, :]
            counterfactual_attention_mask = counterfactual_attention_mask[:, :k, :]
            counterfactual_ctx_mask = counterfactual_ctx_mask[:, :k]

        batch_size, sensitive_seq_len = sensitive_input_ids.size()
        _, num_counterfactual, counterfactual_seq_len = counterfactual_input_ids.size()

        if sensitive_seq_len < 1 or counterfactual_seq_len < 1:
            raise ValueError("CSD requires non-empty sensitive and counterfactual prompts.")

        # Step 1: on-policy rollout in the sensitive context x.
        generation_model = self.accelerator.unwrap_model(model)
        generated_ids = self._run_without_grad(
            model,
            lambda: generation_model.generate(
                input_ids=sensitive_input_ids,
                attention_mask=sensitive_attention_mask,
                do_sample=True,
                temperature=self.csd_rollout_temperature,
                top_p=self.csd_rollout_top_p,
                max_new_tokens=self.csd_rollout_max_new_tokens,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                use_cache=True,
            ),
        )
        if generated_ids.device != sensitive_input_ids.device:
            generated_ids = generated_ids.to(sensitive_input_ids.device)
        generated_y_ids = generated_ids[:, sensitive_seq_len:]
        generated_len = generated_y_ids.size(1)

        if generated_len == 0:
            student_outputs = model(input_ids=sensitive_input_ids, attention_mask=sensitive_attention_mask, return_dict=True)
            loss = student_outputs.logits.sum() * 0.0
            return (loss, student_outputs) if return_outputs else loss

        generated_y_mask = self._response_mask(generated_y_ids).to(sensitive_attention_mask.dtype)

        # Step 2: student forward with gradients on [x ; y].
        student_input_ids = torch.cat([sensitive_input_ids, generated_y_ids], dim=1)
        student_attention_mask = torch.cat([sensitive_attention_mask, generated_y_mask], dim=1)
        student_outputs = self._forward(model, student_input_ids, student_attention_mask)

        # Decoder-only token alignment:
        # logits at position t predict token at t + 1.
        student_logits_y = student_outputs.logits[
            :, sensitive_seq_len - 1 : sensitive_seq_len + generated_len - 1, :
        ]
        student_log_probs = F.log_softmax(student_logits_y.float(), dim=-1)

        # Step 3: teacher forward in K counterfactual contexts x' for the same sampled y.
        expanded_y_ids = generated_y_ids.unsqueeze(1).expand(batch_size, num_counterfactual, generated_len)
        expanded_y_mask = generated_y_mask.unsqueeze(1).expand(batch_size, num_counterfactual, generated_len)
        teacher_input_ids = torch.cat([counterfactual_input_ids, expanded_y_ids], dim=-1)
        teacher_attention_mask = torch.cat([counterfactual_attention_mask, expanded_y_mask], dim=-1)

        merged_teacher_input_ids = teacher_input_ids.reshape(batch_size * num_counterfactual, -1)
        merged_teacher_attention_mask = teacher_attention_mask.reshape(batch_size * num_counterfactual, -1)
        merged_teacher_logits = self._run_without_grad(
            model,
            lambda: self._forward(model, merged_teacher_input_ids, merged_teacher_attention_mask).logits,
        )
        if merged_teacher_logits.device != student_logits_y.device:
            merged_teacher_logits = merged_teacher_logits.to(student_logits_y.device)

        teacher_logits = merged_teacher_logits.view(
            batch_size, num_counterfactual, counterfactual_seq_len + generated_len, -1
        )
        teacher_logits_y = teacher_logits[
            :, :, counterfactual_seq_len - 1 : counterfactual_seq_len + generated_len - 1, :
        ]

        # Step 4: average the teacher distributions over counterfactual contexts.
        teacher_probs = F.softmax(teacher_logits_y.float(), dim=-1)
        counterfactual_ctx_mask = counterfactual_ctx_mask.to(teacher_probs.dtype).unsqueeze(-1).unsqueeze(-1)
        target_probs = (teacher_probs * counterfactual_ctx_mask).sum(dim=1)
        target_probs = target_probs / counterfactual_ctx_mask.sum(dim=1).clamp_min(1.0)
        target_probs = target_probs.clamp_min(1e-8)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Step 5: token-level KL over generated y tokens only.
        token_kl = F.kl_div(student_log_probs, target_probs, reduction="none").sum(dim=-1)
        valid_token_mask = generated_y_mask.to(token_kl.dtype)
        loss = (token_kl * valid_token_mask).sum() / valid_token_mask.sum().clamp_min(1.0)

        return (loss, student_outputs) if return_outputs else loss
