"""Opt-in streaming GEMM controls; no CUDA work or metadata reads here."""

from __future__ import annotations

import os

EXPECTED_M_ENV = "SGLANG_DEEPEP_STREAMING_EXPECTED_M"
STAGES_ENV = "SGLANG_DEEPEP_STREAMING_GEMM_STAGES"


def resolve_expected_m(explicit: int | None) -> int | None:
    """Explicit API hint wins; unset/none preserves the DeepGEMM default.

    The hint is expected rows PER EXPERT, not capacity or total routed rows.
    It only selects a kernel configuration; actual work still follows psums.
    """
    value = explicit
    if value is None:
        raw = os.getenv(EXPECTED_M_ENV, "none").strip().lower()
        if raw in ("", "none"):
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{EXPECTED_M_ENV} must be none or a positive integer") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("expected_m_per_expert must be a positive integer")
    return value


def gemm_stages_enabled() -> bool:
    raw = os.getenv(STAGES_ENV, "0").strip().lower()
    if raw not in ("0", "1"):
        raise ValueError(f"{STAGES_ENV} must be 0 or 1")
    return raw == "1"


def record_boundary(events: dict, name: str, event_factory) -> None:
    event = event_factory(enable_timing=True)
    event.record()
    events[name] = event


def stage_intervals_ms(events: dict) -> dict[str, float]:
    """Consecutive CUDA-event envelopes, including launch gaps/contending work."""
    items = list(events.items())
    return {
        name: previous.elapsed_time(current)
        for (_, previous), (name, current) in zip(items, items[1:])
    }
