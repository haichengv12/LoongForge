# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Replicated-compute ZeRO-1 trainer for LingBot VLA v2."""

import logging
from collections import deque
from contextlib import contextmanager

import torch

from loongforge.embodied.model.lingbot_vla_v2.parameter_policy import (
    LingbotVlaV2ParameterPolicy,
)
from loongforge.embodied.model.lingbot_vla_v2.recipe import LingbotVlaV2Recipe
from loongforge.embodied.optimizer import build_optimizer
from loongforge.embodied.train.trainers.custom.lingbot_vla_v2.zero1 import (
    Zero1CheckpointIO,
    Zero1ParameterManager,
)
from loongforge.embodied.train.trainers.supervised.finetune_trainer import (
    FinetuneTrainer,
)

logger = logging.getLogger(__name__)


class LingbotVlaV2Zero1Trainer(FinetuneTrainer):
    """Own the LingBot recipe and replicated ZeRO-1 lifecycle."""

    def __init__(self, training_args, model_cfg, data_cfg):
        super().__init__(training_args, model_cfg, data_cfg)
        self._recipe = LingbotVlaV2Recipe(model_cfg)
        self._parameter_policy = LingbotVlaV2ParameterPolicy(model_cfg)
        # (batch, teacher handle) pairs fetched during the previous step's optimizer
        # window, consumed in fetch order by ``_fetch_batch``.
        self._pipelined: deque = deque()
        self._teacher_for_batch = None
        self._in_train_step = False
        self._pipeline_teacher = bool(getattr(model_cfg, "pipeline_teacher", True))
        # Dataloader position as of the start of the look-ahead read, i.e. before
        # the batches now sitting in ``_pipelined`` were taken off the loader.
        self._dataloader_state_before_lookahead = None

    def _train_forward(self, batch):
        return self._recipe.forward(self, batch, self._teacher_for_batch)

    def _finalize(self):
        self._teacher_for_batch = None
        self._recipe.close()
        try:
            # The final checkpoint is written inside ``super()._finalize()``, so the
            # look-ahead bookkeeping has to stay intact until it has been saved.
            return super()._finalize()
        finally:
            self._pipelined.clear()
            self._dataloader_state_before_lookahead = None

    def _fetch_batch(self, dl_name: str):
        """Serve a batch whose teachers were already started, when one is queued.

        Only during a training step: evaluation reads the same loader name, and
        handing it a queued training batch would silently train and evaluate on
        different data.
        """
        if self._in_train_step and dl_name == "vla" and self._pipelined:
            batch, self._teacher_for_batch = self._pipelined.popleft()
            return batch
        self._teacher_for_batch = None
        return super()._fetch_batch(dl_name)

    def _start_next_step_teachers(self) -> None:
        """Fetch the next step's batches and start their teachers, here in the
        optimizer window.

        Measured on 8 GPUs: this window absorbs 79% of the teacher's 145.7 ms
        against 34% when the teacher runs into the forward, because 375.9 of its
        389.4 ms is NCCL and it therefore has slack the forward does not.

        Requires the async runner: without it ``submit_teacher`` blocks, which would
        put the whole teacher inside the window instead of alongside it.
        """
        if not self._pipeline_teacher or self._pipelined:
            return
        if not self._recipe.batch_enricher.has_runner:
            return
        # The look-ahead read pulls the next step's batches off the loader, so from
        # here on the loader sits ahead of the training position. A checkpoint
        # written at the end of this step must resume from *this* position, not the
        # advanced one, or the resumed run skips the prefetched batches.
        self._dataloader_state_before_lookahead = super()._get_dataloader_state()
        for _ in range(self.training_args.gradient_accumulation_steps):
            batch = super()._fetch_batch("vla")
            self._pipelined.append((batch, self._recipe.submit_teacher(batch)))

    def _get_dataloader_state(self):
        """Report the training position, not the look-ahead read position."""
        if self._pipelined and self._dataloader_state_before_lookahead is not None:
            return self._dataloader_state_before_lookahead
        return super()._get_dataloader_state()

    def _wrap_model_for_training(self) -> None:
        self._recipe.setup_parallel_state(self)
        policy = self.model.policy.to(self.ctx.device)
        if self.model_cfg.gradient_checkpointing and hasattr(
            policy, "gradient_checkpointing_enable"
        ):
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.policy = policy
        self._zero1 = Zero1ParameterManager(
            policy,
            group=None,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            parameter_policy=self._parameter_policy,
            grad_reduce_dtype=self.model_cfg.grad_reduce_dtype,
            param_sync_precision=self.model_cfg.param_sync_precision,
            param_sync_fp8_block=self.model_cfg.param_sync_fp8_block,
            param_sync_fp8_reprime_interval=self.model_cfg.param_sync_fp8_reprime_interval,
            grad_overlap=self.model_cfg.grad_overlap,
            param_overlap=self.model_cfg.param_overlap,
            bucket_mb=self.model_cfg.comm_bucket_mb,
            grad_inflight_mb=self.model_cfg.grad_inflight_mb,
        )
        self._optimizer_parameter_model = self._zero1.optimizer_view()
        if self.ctx.is_main:
            logger.info(
                "ZeRO-1 collective precision: grad_reduce=%s parameter_sync=%s "
                "(router/gate and 1-D tensors stay fp32)",
                self._zero1.grad_reduce_mode,
                self._zero1.param_sync_precision,
            )
            logger.info(
                "ZeRO-1 overlap: grad=%s param=%s bucket=%dMiB",
                self.model_cfg.grad_overlap,
                self.model_cfg.param_overlap,
                self._zero1.bucket_mb,
            )
        self._recipe.setup(self)
        if self.ctx.is_main:
            logger.info(
                "Teacher targets start %s",
                "one step early, in the optimizer window"
                if self._pipeline_teacher
                else "at the head of their own step",
            )

    def _should_sync_grads(self, micro: int, grad_accum: int) -> bool:
        sync = super()._should_sync_grads(micro, grad_accum)
        if micro == grad_accum - 1:
            self._zero1.begin_gradient_sync()
        return sync

    @contextmanager
    def _grad_sync_ctx(self, sync_grads: bool):
        """Let accumulation micro-steps run without any wrapper-level gating.

        The base implementation gates through the model wrapper: ``no_sync()`` for
        DDP, otherwise ``set_requires_gradient_sync`` for FSDP2. The replicated
        ZeRO-1 model is neither, so it has no such method and any
        ``grad_accum > 1`` run used to die with an ``AttributeError`` before the
        first optimizer step.

        No gating is needed here. Backward only accumulates into ``.grad``, and
        the reduction is armed exactly once per optimizer step by
        ``_should_sync_grads`` on the last micro-step, so the accumulate-then-
        reduce-once semantics already hold.
        """
        del sync_grads
        yield

    def _run_forward_backward_block(self):
        result = super()._run_forward_backward_block()
        self._zero1.finish_gradient_sync()
        return result

    def _train_step(self):
        self._zero1.begin_parameter_sync()
        self._in_train_step = True
        try:
            result = super()._train_step()
        finally:
            self._in_train_step = False
        self._zero1.finish_parameter_sync()
        return result

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer = self._recipe.build_optimizer(self)
        if optimizer is None:
            optimizer = build_optimizer(self.model, self.training_args)
        self.optimizer = optimizer
        # The shared checkpoint backend owns the file layout and the aux files;
        # this object owns the rank-local FP32 masters and Muon/AdamW state.
        self.optimizer.zero_checkpoint_io = Zero1CheckpointIO(self._zero1, optimizer)
        self._recipe.wire_parameter_sync(self)
        return optimizer

    def _clip_gradients(self, max_norm: float) -> float:
        norm = self._zero1.clip_grad_norm_(max_norm)
        # The muon + adamw + parameter_sync region starts here and is mostly NCCL,
        # so its tensor cores sit idle. Fill it with the next step's teachers.
        self._start_next_step_teachers()
        return norm

    def _clean_nan_gradients(self):
        for parameter in self._zero1.master.values():
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def _save_checkpoint(self):
        # The fp32 masters and their optimizer state are rank-local, and only the
        # dcp path keeps per-rank files. A legacy-format training-state save would
        # map them through the model's parameter FQNs and silently write rank0's
        # shard alone, so refuse it rather than emit a checkpoint that only looks
        # resumable.
        if self.training_args.save_training_state and (
            self.training_args.use_lora or self.training_args.save_format != "dcp"
        ):
            raise ValueError(
                "ZeRO-1 training-state checkpoints require --save-format=dcp; "
                "pass --no-save-training-state to export weights only."
            )
        super()._save_checkpoint()


__all__ = ["LingbotVlaV2Zero1Trainer"]
