from __future__ import annotations


def compress_memories(contents: list[str], max_chars: int = 1800) -> str:
    lines: list[str] = []
    used = 0
    for item in contents:
        text = " ".join(str(item).split())
        if not text:
            continue
        room = max_chars - used
        if room <= 0:
            break
        if len(text) > room:
            text = text[: max(0, room - 1)] + "…"
        lines.append(f"- {text}")
        used += len(text) + 3
    return "\n".join(lines)
