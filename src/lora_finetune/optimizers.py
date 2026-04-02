"""Custom optimizer helpers for non-standard trainer integrations."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

_MUON_NAME_BLOCKLIST = ("embed", "embedding", "norm", "ln", "bias")


def is_muon_available() -> bool:
    """Return whether the active torch build exposes torch.optim.Muon."""
    return hasattr(torch.optim, "Muon")


def _is_muon_eligible_parameter(name: str, parameter: Tensor) -> bool:
    normalized_name = name.lower()
    if any(token in normalized_name for token in _MUON_NAME_BLOCKLIST):
        return False
    return parameter.ndim == 2


def partition_muon_parameters(
    model: torch.nn.Module,
) -> Tuple[List[Tuple[str, Tensor]], List[Tuple[str, Tensor]]]:
    """Split trainable parameters into Muon-eligible and AdamW fallback sets."""
    muon_parameters: List[Tuple[str, Tensor]] = []
    adamw_parameters: List[Tuple[str, Tensor]] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_muon_eligible_parameter(name, parameter):
            muon_parameters.append((name, parameter))
        else:
            adamw_parameters.append((name, parameter))

    return muon_parameters, adamw_parameters


class HybridMuonAdamW(Optimizer):
    """Optimizer wrapper that delegates updates to Muon and AdamW child optimizers."""

    def __init__(
        self,
        *,
        muon_optimizer: Optional[Optimizer],
        adamw_optimizer: Optional[Optimizer],
    ) -> None:
        self.muon_optimizer = muon_optimizer
        self.adamw_optimizer = adamw_optimizer
        self.optimizers = [opt for opt in (muon_optimizer, adamw_optimizer) if opt is not None]
        if not self.optimizers:
            raise ValueError("HybridMuonAdamW requires at least one child optimizer")

        param_groups = []
        defaults = {}
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
            defaults.update(getattr(optimizer, "defaults", {}))

        super().__init__(param_groups, defaults)
        self.param_groups = param_groups
        self.defaults = defaults

    @property
    def state(self):
        merged_state = {}
        for optimizer in self.optimizers:
            optimizer_state = getattr(optimizer, "state", None)
            if isinstance(optimizer_state, dict):
                merged_state.update(optimizer_state)
        return merged_state

    @state.setter
    def state(self, value) -> None:
        self._state = value

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "state": self.state,
            "param_groups": self.param_groups,
            "muon_optimizer": (
                None if self.muon_optimizer is None else self.muon_optimizer.state_dict()
            ),
            "adamw_optimizer": (
                None if self.adamw_optimizer is None else self.adamw_optimizer.state_dict()
            ),
        }

    def load_state_dict(self, state_dict):
        if self.muon_optimizer is not None and state_dict.get("muon_optimizer") is not None:
            self.muon_optimizer.load_state_dict(state_dict["muon_optimizer"])
        if self.adamw_optimizer is not None and state_dict.get("adamw_optimizer") is not None:
            self.adamw_optimizer.load_state_dict(state_dict["adamw_optimizer"])
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        return None
