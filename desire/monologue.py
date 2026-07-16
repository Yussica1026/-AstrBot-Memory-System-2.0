from __future__ import annotations

from .core import DesireState


def render_monologue(state: DesireState) -> str:
    if state.thoughts:
        return state.thoughts[-1]
    strongest = max(state.drives.items(), key=lambda item: item[1])
    return f"当前最强驱动：{strongest[0]}={strongest[1]:.2f}"
