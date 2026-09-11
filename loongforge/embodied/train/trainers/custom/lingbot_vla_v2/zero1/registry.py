# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Stable parameter metadata for replicated-compute ZeRO-1.

The registry runs the setup in explicitly ordered phases. The previous version
depended on statement order inside one constructor: the compute dtype was
rewritten in the same loop that created the fp32 masters, while the wire dtypes
were resolved later from whatever dtype the compute parameter happened to hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from loongforge.embodied.train.trainers.custom.lingbot_vla_v2.zero1.layout import (
    ShardingSpec,
)
from loongforge.embodied.train.trainers.custom.lingbot_vla_v2.zero1.ownership import (
    OwnershipPlanner,
    ParameterOwnership,
)

_GRAD_REDUCE_MODES = ("fp32", "bf16", "mixed", "compute")
_PARAM_SYNC_PRECISIONS = ("fp32", "bf16", "bf16_ef", "bf16_ef_delta", "fp8_e4m3_delta")
# Each parameter-publish precision decomposes into the three internal levers the
# collective paths consume: the fp32-master publish dtype, the bf16 error-feedback
# variant, and the delta quantization. Restricting the public surface to these five
# named combinations makes conflicting settings (bf16 and fp8 at once) unrepresentable.
_PARAM_SYNC_DECOMPOSITION = {
    "fp32": ("compute", "none", "none"),
    "bf16": ("bf16", "none", "none"),
    "bf16_ef": ("bf16", "error_feedback", "none"),
    "bf16_ef_delta": ("bf16", "error_feedback_delta", "none"),
    "fp8_e4m3_delta": ("compute", "none", "fp8_e4m3_delta"),
}


@dataclass
class ParameterRecord:
    """Everything the collective paths need to know about one parameter."""

    name: str
    compute: nn.Parameter
    ownership: ParameterOwnership
    spec: ShardingSpec
    numel: int
    optimizer_kind: str
    is_comm_critical: bool
    master: nn.Parameter | None = None
    shard_start: int = 0
    shard_count: int = 0
    grad_wire_dtype: torch.dtype = torch.float32
    param_wire_dtype: torch.dtype = torch.float32
    reverse_position: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full (unsharded) parameter shape."""
        return self.ownership.shape

    @property
    def is_sharded(self) -> bool:
        """True when the fp32 master is split along dim 0."""
        return self.spec.is_sharded

    @property
    def owner(self) -> int | None:
        """Return the owning rank, or ``None`` for a sharded parameter."""
        return self.spec.owner

    @property
    def master_bytes(self) -> int:
        """Return the fp32 master footprint used for bucket accounting."""
        return self.numel * 4


class ParameterRegistry:
    """Own the compute replicas, the fp32 masters and the resolved wire dtypes."""

    def __init__(
        self,
        module,
        rank,
        world_size,
        parameter_policy,
        grad_reduce_dtype=None,
        param_sync_precision=None,
        param_sync_fp8_block=256,
        param_sync_fp8_reprime_interval=0,
    ):
        """Enumerate, plan, cast, create masters and resolve dtypes, in that order."""
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.policy = parameter_policy
        self.grad_reduce_mode = self._validate(
            "grad_reduce_dtype", grad_reduce_dtype or "fp32", _GRAD_REDUCE_MODES
        )
        self.param_sync_precision = self._validate(
            "param_sync_precision",
            param_sync_precision or "fp32",
            _PARAM_SYNC_PRECISIONS,
        )
        # Fan the single public knob out into the internal levers the collective
        # paths read. param_sync_mode still governs the VLM (bf16-compute) publish;
        # only the fp32 action-expert masters see the compensation/quantization.
        (
            self.param_sync_mode,
            self.param_sync_compensation,
            self.param_sync_quantization,
        ) = _PARAM_SYNC_DECOMPOSITION[self.param_sync_precision]
        self.param_sync_fp8_block = int(param_sync_fp8_block)
        if (
            self.param_sync_fp8_block <= 0
            or self.param_sync_fp8_block & (self.param_sync_fp8_block - 1)
            or self.param_sync_fp8_block > 1 << 20
        ):
            raise ValueError(
                "param_sync_fp8_block must be a positive power of two <= 1048576"
            )
        self.param_sync_fp8_reprime_interval = int(param_sync_fp8_reprime_interval)
        if self.param_sync_fp8_reprime_interval < 0:
            raise ValueError("param_sync_fp8_reprime_interval must be non-negative")

        # Phase 1: enumerate trainable parameters in registration order.
        self.compute = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        # Phase 2: plan ownership before anything mutates the parameters.
        self.ownership, specs = OwnershipPlanner(parameter_policy, world_size).plan(
            self.compute.items()
        )
        self.specs = {item.name: item for item in self.ownership}

        # Phase 3: classify precision-critical parameters and optimizer kind from
        # the original tensors, before any dtype is rewritten.
        self.records: list[ParameterRecord] = []
        for index, item in enumerate(self.ownership):
            compute = self.compute[item.name]
            self.records.append(
                ParameterRecord(
                    name=item.name,
                    compute=compute,
                    ownership=item,
                    spec=specs[item.name],
                    numel=compute.numel(),
                    optimizer_kind=parameter_policy.optimizer_kind(item.name, compute),
                    is_comm_critical=parameter_policy.is_comm_precision_critical(
                        item.name, compute
                    ),
                    # named_parameters() follows the forward pass, so its reverse
                    # order approximates the order gradients become ready in backward.
                    reverse_position=-index,
                )
            )
        self._by_name = {record.name: record for record in self.records}

        # Phase 4: create the fp32 masters from the pre-cast values, then apply the
        # policy compute dtype. This order is load bearing: a master built after a
        # bf16 downcast would lose the bits the optimizer needs.
        self.master: dict[str, nn.Parameter] = {}
        for record in self.records:
            self._create_master(record)
            self._apply_compute_dtype(record)

        # Phase 5: resolve wire dtypes now that every compute dtype is final.
        for record in self.records:
            record.grad_wire_dtype = self._resolve_grad_wire_dtype(record)
            record.param_wire_dtype = self._resolve_param_wire_dtype(record)

    @staticmethod
    def _validate(name, value, allowed):
        """Reject an unsupported precision mode instead of falling back silently."""
        lowered = str(value).lower()
        if lowered not in allowed:
            raise ValueError(
                f"invalid {name}={lowered!r}; expected one of {', '.join(allowed)}"
            )
        return lowered

    def _create_master(self, record):
        """Create this rank's fp32 master shard, if it owns one."""
        compute = record.compute
        if record.is_sharded:
            start, count = record.spec.local_range(self.rank)
            record.shard_start = start
            record.shard_count = count
            value = compute.detach()[start : start + count].float().clone()
        elif record.owner == self.rank:
            value = compute.detach().float().clone()
        else:
            return
        record.master = nn.Parameter(value, requires_grad=True)
        self.master[record.name] = record.master

    def _apply_compute_dtype(self, record):
        """Cast the compute replica to the dtype the policy asks for."""
        compute_dtype = self.policy.compute_dtype(record.name, record.compute)
        if record.compute.dtype != compute_dtype:
            record.compute.data = record.compute.data.to(compute_dtype)

    def _resolve_grad_wire_dtype(self, record):
        """Pick the wire dtype for one parameter's gradient reduction.

        Parameters the policy marks precision-critical (MoE router/gate, 1-D
        tensors) always travel at full precision, and buckets are split by dtype
        so a critical parameter can never be dragged into a downcast payload.
        """
        mode = self.grad_reduce_mode
        if mode == "fp32" or record.is_comm_critical:
            return torch.float32
        if record.is_sharded:
            # Sharded stacks only downcast under the explicit bf16 mode; the
            # "mixed"/"compute" modes keep the reduce-scatter in fp32.
            return torch.bfloat16 if mode == "bf16" else torch.float32
        if mode in ("bf16", "mixed"):
            return torch.bfloat16
        if mode == "compute":
            return (
                torch.bfloat16
                if record.compute.dtype == torch.bfloat16
                else torch.float32
            )
        return torch.float32

    def _resolve_param_wire_dtype(self, record):
        """Pick the wire dtype for publishing one updated parameter to replicas."""
        if record.is_comm_critical:
            return torch.float32
        compute_dtype = record.compute.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            return compute_dtype
        # An fp32 compute parameter can still be published in bf16: the fp32
        # master keeps the update precision, replicas just receive a rounded
        # copy -- and all ranks, owner included, read the same rounded bytes
        # back, so the replicas stay bit-identical.
        if (
            self.param_sync_quantization == "fp8_e4m3_delta"
            and not record.is_comm_critical
            and not record.is_sharded
            and record.compute.dtype == torch.float32
        ):
            return torch.uint8
        if self.param_sync_mode == "bf16":
            return torch.bfloat16
        return torch.float32

    def record(self, name) -> ParameterRecord:
        """Return the record registered under ``name``."""
        return self._by_name[name]

    @property
    def device(self):
        """Return the device the compute replicas live on."""
        return next(iter(self.compute.values())).device

    def named_master_parameters(self):
        """Return the ``(name, master parameter)`` pairs owned by this rank."""
        return list(self.master.items())

    def whole_tensor_records(self):
        """Return the whole-tensor records, in registration order."""
        return [record for record in self.records if not record.is_sharded]

    def dim0_sharded_records(self):
        """Return the dim-0 sharded records, in registration order."""
        return [record for record in self.records if record.is_sharded]


class MasterParameterView:
    """Minimal module-like view consumed by the vendored optimizer builder."""

    def __init__(self, named_parameters):
        """Store the ``(name, parameter)`` pairs this view exposes."""
        self._named_parameters = tuple(named_parameters)

    def named_parameters(self):
        """Yield the ``(name, parameter)`` pairs this view was built from."""
        return iter(self._named_parameters)

    def named_modules(self):
        """Yield nothing; the view has no module tree."""
        return iter(())


__all__ = ["MasterParameterView", "ParameterRecord", "ParameterRegistry"]
