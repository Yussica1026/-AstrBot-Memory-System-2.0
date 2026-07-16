from __future__ import annotations

from datetime import datetime


def half_month_key(dt: datetime) -> str:
    half = "H1" if dt.day <= 15 else "H2"
    return f"{dt.year:04d}-{dt.month:02d}-{half}"


def should_run_archive(now: datetime) -> bool:
    return now.day in {1, 16} and now.hour == 4
