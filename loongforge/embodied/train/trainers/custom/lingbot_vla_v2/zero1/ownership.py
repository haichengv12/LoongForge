# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Owner assignment for replicated-compute ZeRO-1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn

from loongforge.embodied.train.trainers.custom.lingbot_vla_v2.zero1.layout import (
    DIM0_RANGE,
    WHOLE_TENSOR,
    ShardingSpec,
)


@dataclass(frozen=True)
class ParameterOwnership:
    """Which rank owns a parameter, and how experts are split when sharded.

    Kept as the on-the-wire ownership schema: ``state_dict`` serializes these
    fields, so renaming or reordering them would invalidate the checkpoint
    compatibility check. Internal code branches on ``ShardingSpec.mode``.
    """

    name: str
    shape: tuple[int, ...]
    owner: int | None
    expert_counts: tuple[int, ...] = ()

    @property
    def is_dim0_sharded(self) -> bool:
        """True when the parameter has no single owner and is split along dim 0."""
        return self.owner is None

    def sharding_spec(self) -> ShardingSpec:
        """Return the explicit layout this ownership record describes."""
        if self.owner is None:
            return ShardingSpec(DIM0_RANGE, None, self.expert_counts)
        return ShardingSpec(WHOLE_TENSOR, self.owner)


def assign_parameter_owners(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    world_size: int,
    parameter_policy=None,
):
    """Assign full tensors to owners and policy-approved expert shards on dim 0."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if parameter_policy is None:
        raise ValueError("parameter_policy is required")
    policy = parameter_policy
    items = list(named_parameters)
    loads = [0] * world_size
    result = []
    for name, parameter in items:
        shape = tuple(parameter.shape)
        if policy.is_expert_shard(name, parameter):
            base, remainder = divmod(shape[0], world_size)
            counts = tuple(base + (rank < remainder) for rank in range(world_size))
            for rank, count in enumerate(counts):
                loads[rank] += count * parameter[0].numel() * 4 if shape[0] else 0
            result.append(ParameterOwnership(name, shape, None, counts))
        else:
            owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
            loads[owner] += parameter.numel() * 4
            result.append(ParameterOwnership(name, shape, owner))
    return result


class OwnershipPlanner:
    """Turn named parameters into ownership records and explicit sharding specs.

    Only the validated greedy scheme is implemented. Reordering owners changes
    Muon's same-shape megabatch grouping (measured up to 2% gradient-norm drift),
    so an alternative balancing scheme belongs behind a new mode, never as a
    tweak here.
    """

    def __init__(self, parameter_policy, world_size):
        """Record the policy that decides sharding and the target world size."""
        if parameter_policy is None:
            raise ValueError("parameter_policy is required")
        self._policy = parameter_policy
        self._world_size = int(world_size)

    def plan(self, named_parameters):
        """Return ``(ownership_records, {name: ShardingSpec})``."""
        ownership = assign_parameter_owners(
            named_parameters, self._world_size, self._policy
        )
        return ownership, {item.name: item.sharding_spec() for item in ownership}


__all__ = ["OwnershipPlanner", "ParameterOwnership", "assign_parameter_owners"]
