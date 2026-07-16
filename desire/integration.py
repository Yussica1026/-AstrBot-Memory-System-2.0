from __future__ import annotations

import json

from .core import DesireState


def serialize_state(state: DesireState) -> str:
    return json.dumps({"drives": state.drives, "thoughts": state.thoughts}, ensure_ascii=False)


def deserialize_state(raw: str) -> DesireState:
    data = json.loads(raw)
    state = DesireState(drives=data.get("drives", {}), thoughts=data.get("thoughts", []))
    state.clamp()
    return state
