from __future__ import annotations

from .core import DesireState


def add_thought(state: DesireState, thought: str) -> None:
    text = thought.strip()
    if text:
        state.thoughts.append(text)
