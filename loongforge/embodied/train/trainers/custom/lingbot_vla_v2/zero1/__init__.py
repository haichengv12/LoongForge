# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ZeRO-1 parameter ownership and master-weight management for LingBot VLA v2."""
from .checkpoint_io import Zero1CheckpointIO
from .gradient_reducer import GradientReducer, GradientSyncEntry
from .layout import DIM0_RANGE, WHOLE_TENSOR, BufferPool, ShardingSpec
from .optimizer_adapter import OptimizerAdapter
from .ownership import OwnershipPlanner, ParameterOwnership, assign_parameter_owners
from .parameter_manager import Zero1ParameterManager
from .parameter_sync import ParameterSynchronizer, ParameterSyncEntry
from .registry import MasterParameterView, ParameterRecord, ParameterRegistry

__all__ = [
    "DIM0_RANGE",
    "WHOLE_TENSOR",
    "BufferPool",
    "GradientReducer",
    "GradientSyncEntry",
    "MasterParameterView",
    "OptimizerAdapter",
    "OwnershipPlanner",
    "ParameterOwnership",
    "ParameterRecord",
    "ParameterRegistry",
    "ParameterSyncEntry",
    "ParameterSynchronizer",
    "ShardingSpec",
    "Zero1CheckpointIO",
    "Zero1ParameterManager",
    "assign_parameter_owners",
]
