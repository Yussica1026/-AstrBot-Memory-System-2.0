from __future__ import annotations

from .core import DesireState


def tick(state: DesireState) -> DesireState:
    for key in state.drives:
        state.drives[key] += 0.01
    state.clamp()
    return state
