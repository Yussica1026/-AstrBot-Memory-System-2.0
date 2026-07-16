from __future__ import annotations

from .core import DesireState


def apply_safety_valve(state: DesireState) -> DesireState:
    state.clamp()
    return state
