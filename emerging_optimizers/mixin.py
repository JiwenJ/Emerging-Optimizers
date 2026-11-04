# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
from typing import Any, Literal, override

import torch


WeightDecayT = Literal["decoupled", "independent", "l2", "palm"]


class WeightDecayMixin:
    """Mixin for weight decay

    Supports different types of weight decay:

    - "decoupled": weight decay is applied directly to params without changing gradients
    - "independent": similar as decoupled weight decay, but without tying weight decay and learning rate
    - "l2": classic L2 regularization
    - "palm": weight decay is applied directly to params with lr^2 scaling
    """

    def _apply_weight_decay_inplace(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Depends on the weight decay option, p or grad will be updated in place"""
        if weight_decay == 0.0:
            return

        weight_decay_method = getattr(self, "weight_decay_method", "l2")
        if weight_decay_method == "decoupled":
            p.add_(p, alpha=(-weight_decay * lr))
        elif weight_decay_method == "independent":
            p.add_(p, alpha=-weight_decay)
        elif weight_decay_method == "l2":
            grad.add_(p, alpha=weight_decay)
        elif weight_decay_method == "palm":
            p.add_(p, alpha=(-(weight_decay * lr * lr)))
        else:
            raise ValueError(f"Invalid weight decay method: {weight_decay_method}")


class PreserveStateDtypeMixin(torch.optim.Optimizer):
    """Optimizer base that keeps the saved state dtypes in :meth:`load_state_dict`.

    ``torch.optim.Optimizer.load_state_dict`` casts floating point state to the parameter dtype, which turns the fp32
    state these optimizers keep for bf16/fp16 parameters into low precision and breaks the next ``step()``.
    """

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        saved_ids = chain.from_iterable(group["params"] for group in state_dict["param_groups"])
        params = chain.from_iterable(group["params"] for group in self.param_groups)
        for saved_id, p in zip(saved_ids, params):
            if saved_id not in state_dict["state"]:
                continue
            state = self.state[p]
            for key, saved_value in state_dict["state"][saved_id].items():
                state[key] = _with_saved_dtype(state[key], saved_value)


def _with_saved_dtype(value: Any, saved_value: Any) -> Any:
    if isinstance(value, torch.Tensor) and isinstance(saved_value, torch.Tensor):
        if value.is_floating_point() and value.dtype != saved_value.dtype:
            return saved_value.to(device=value.device, copy=True)
        return value
    if isinstance(value, (list, tuple)) and isinstance(saved_value, (list, tuple)) and len(value) == len(saved_value):
        return type(value)(_with_saved_dtype(v, s) for v, s in zip(value, saved_value))
    return value
