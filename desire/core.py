from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_DRIVES = {
    "attachment": 0.5,
    "curiosity": 0.5,
    "security": 0.5,
    "expression": 0.5,
    "rest": 0.5,
    "achievement": 0.5,
    "intimacy": 0.5,
    "autonomy": 0.5,
    "play": 0.5,
}


@dataclass
class DesireState:
    drives: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DRIVES))
    thoughts: list[str] = field(default_factory=list)

    def clamp(self) -> None:
        for key, value in list(self.drives.items()):
            self.drives[key] = max(0.0, min(1.0, float(value)))
