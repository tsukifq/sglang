"""Strict opt-in, request-local MoE CUDA timeline helpers.

The timeline is carried in a context variable so dispatcher, runner, and
kernel-adapter code can record events without changing production call
signatures.  No event is created unless the outer FusedMoE probe selects the
current layer invocation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import os
import queue
import socket
import threading
import traceback
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

_ACTIVE_MOE_TIMELINE: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "sglang_active_moe_timeline", default=None
)
_COLLECTOR_LOCK = threading.Lock()
_COLLECTOR_QUEUE: Optional[queue.Queue[Callable[[], None]]] = None
_EVENT_GUARD_ENV = "SGLANG_DEEPEP_TIMELINE_EVENT_GUARD_NS"

MOE_COMPONENT_PROFILE_SCHEMA = "async-moe-component-profile-v1"
_EXECUTION_MODELS = frozenset(("staged", "streaming"))
_PROFILE_DETAILS = frozenset(("summary", "component", "lane", "hardware"))


def canonical_moe_profile_detail(detail: str, execution_model: str) -> str:
    """Map legacy timeline detail names onto the shared profiler contract."""

    if execution_model not in _EXECUTION_MODELS:
        raise ValueError(f"unsupported MoE execution model: {execution_model!r}")
    normalized = detail.strip().lower()
    if normalized == "arrival":
        return "summary"
    if normalized == "full":
        return "lane" if execution_model == "streaming" else "component"
    if normalized not in _PROFILE_DETAILS:
        raise ValueError(f"unsupported MoE profile detail: {detail!r}")
    return normalized


def _normalize_profile_event(name: str, point: Any) -> dict[str, Any]:
    if isinstance(point, Mapping):
        normalized = dict(point)
        if "rank_local_ms" not in normalized:
            raise ValueError(f"profile event {name!r} lacks rank_local_ms")
    else:
        normalized = {"rank_local_ms": point}
    rank_local_ms = float(normalized["rank_local_ms"])
    if not math.isfinite(rank_local_ms):
        raise ValueError(f"profile event {name!r} has a non-finite timestamp")
    normalized["rank_local_ms"] = rank_local_ms
    return normalized


def _profile_interval_ms(
    events: Mapping[str, Mapping[str, Any]], start: str, end: str
) -> Optional[float]:
    if start not in events or end not in events:
        return None
    start_ms = float(events[start]["rank_local_ms"])
    end_ms = float(events[end]["rank_local_ms"])
    if end_ms < start_ms:
        raise ValueError(f"profile event {end!r} precedes {start!r}")
    return end_ms - start_ms


def _profile_overlap_ms(
    events: Mapping[str, Mapping[str, Any]],
    left_start: str,
    left_end: str,
    right_start: str,
    right_end: str,
) -> Optional[float]:
    if not {left_start, left_end, right_start, right_end}.issubset(events):
        return None
    left = (
        float(events[left_start]["rank_local_ms"]),
        float(events[left_end]["rank_local_ms"]),
    )
    right = (
        float(events[right_start]["rank_local_ms"]),
        float(events[right_end]["rank_local_ms"]),
    )
    if left[1] < left[0] or right[1] < right[0]:
        raise ValueError("profile interval end precedes its start")
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def build_moe_component_profile(
    *,
    execution_model: str,
    detail: str,
    events: Mapping[str, Any],
    event_provenance: Optional[Mapping[str, str]] = None,
    counters: Optional[Mapping[str, Any]] = None,
    items: Optional[Sequence[Mapping[str, Any]]] = None,
    capabilities: Optional[Mapping[str, bool]] = None,
) -> dict[str, Any]:
    """Build the common staged/streaming MoE profiler payload.

    The contract is deliberately a DAG rather than a list of mutually
    exclusive stages.  Streaming dispatch, compute, and combine envelopes may
    overlap.  Component adapters may omit events they cannot observe exactly;
    proxy events must use distinct names and declare their provenance.
    """

    canonical_detail = canonical_moe_profile_detail(detail, execution_model)
    normalized_events = {
        name: _normalize_profile_event(name, point) for name, point in events.items()
    }
    if "layer_entry" not in normalized_events:
        raise ValueError("MoE component profile requires layer_entry")
    if "layer_output_ready" not in normalized_events:
        raise ValueError("MoE component profile requires layer_output_ready")

    critical_path_ms = _profile_interval_ms(
        normalized_events, "layer_entry", "layer_output_ready"
    )
    assert critical_path_ms is not None
    metrics: dict[str, Any] = {"critical_path_ms": critical_path_ms}

    dispatch_input_ready_ms = _profile_interval_ms(
        normalized_events, "layer_entry", "dispatch_input_ready"
    )
    dispatch_first_ms = _profile_interval_ms(
        normalized_events, "layer_entry", "dispatch_first_output_ready"
    )
    dispatch_all_ms = _profile_interval_ms(
        normalized_events, "layer_entry", "dispatch_all_output_ready"
    )
    dispatch_spread_ms = _profile_interval_ms(
        normalized_events,
        "dispatch_first_output_ready",
        "dispatch_all_output_ready",
    )
    if any(
        value is not None
        for value in (
            dispatch_input_ready_ms,
            dispatch_first_ms,
            dispatch_all_ms,
            dispatch_spread_ms,
        )
    ):
        metrics["dispatch"] = {
            "layer_entry_to_input_ready_ms": dispatch_input_ready_ms,
            "input_to_first_output_ms": dispatch_first_ms,
            "input_to_all_output_ms": dispatch_all_ms,
            "output_readiness_spread_ms": dispatch_spread_ms,
        }
        if "dispatch_input_ready" in normalized_events:
            metrics["dispatch"].update(
                {
                    "ready_to_first_output_ms": _profile_interval_ms(
                        normalized_events,
                        "dispatch_input_ready",
                        "dispatch_first_output_ready",
                    ),
                    "ready_to_all_output_ms": _profile_interval_ms(
                        normalized_events,
                        "dispatch_input_ready",
                        "dispatch_all_output_ready",
                    ),
                }
            )

    consumer_first_ms = _profile_interval_ms(
        normalized_events, "layer_entry", "dispatch_first_consumer_start"
    )
    consumer_all_ms = _profile_interval_ms(
        normalized_events, "layer_entry", "dispatch_all_consumer_start"
    )
    consumer_spread_ms = _profile_interval_ms(
        normalized_events,
        "dispatch_first_consumer_start",
        "dispatch_all_consumer_start",
    )
    if any(
        value is not None
        for value in (consumer_first_ms, consumer_all_ms, consumer_spread_ms)
    ):
        dispatch_metrics = metrics.setdefault("dispatch", {})
        dispatch_metrics.update(
            {
                "input_to_first_consumer_start_ms": consumer_first_ms,
                "input_to_all_consumers_started_ms": consumer_all_ms,
                "consumer_start_spread_ms": consumer_spread_ms,
            }
        )
        if "dispatch_input_ready" in normalized_events:
            dispatch_metrics.update(
                {
                    "ready_to_first_consumer_start_ms": _profile_interval_ms(
                        normalized_events,
                        "dispatch_input_ready",
                        "dispatch_first_consumer_start",
                    ),
                    "ready_to_all_consumers_started_ms": _profile_interval_ms(
                        normalized_events,
                        "dispatch_input_ready",
                        "dispatch_all_consumer_start",
                    ),
                }
            )

    compute_ms = _profile_interval_ms(
        normalized_events, "compute_first_start", "compute_all_done"
    )
    if compute_ms is not None:
        metrics["compute"] = {"envelope_ms": compute_ms}

    combine_ms = _profile_interval_ms(
        normalized_events, "combine_first_start", "combine_all_done"
    )
    if combine_ms is not None:
        metrics["combine"] = {"envelope_ms": combine_ms}

    dispatch_compute_overlap_ms = _profile_overlap_ms(
        normalized_events,
        (
            "dispatch_input_ready"
            if "dispatch_input_ready" in normalized_events
            else "layer_entry"
        ),
        "dispatch_all_output_ready",
        "compute_first_start",
        "compute_all_done",
    )
    compute_combine_overlap_ms = _profile_overlap_ms(
        normalized_events,
        "compute_first_start",
        "compute_all_done",
        "combine_first_start",
        "combine_all_done",
    )
    if dispatch_compute_overlap_ms is not None or compute_combine_overlap_ms is not None:
        metrics["overlap"] = {
            "dispatch_compute_ms": dispatch_compute_overlap_ms,
            "compute_combine_ms": compute_combine_overlap_ms,
        }

    provenance = dict(event_provenance or {})
    unknown_provenance = set(provenance) - set(normalized_events)
    if unknown_provenance:
        raise ValueError(
            "event provenance references missing events: "
            + ", ".join(sorted(unknown_provenance))
        )

    normalized_items = [dict(item) for item in (items or ())]
    observed_capabilities = {
        "exact_dispatch_input_ready": "dispatch_input_ready" in normalized_events,
        "exact_dispatch_output_ready": {
            "dispatch_first_output_ready",
            "dispatch_all_output_ready",
        }.issubset(normalized_events),
        "per_work_item": bool(normalized_items),
        "physical_bytes": False,
        "hardware_counters": False,
    }
    observed_capabilities.update(capabilities or {})

    normalized_counters = dict(counters or {})
    dispatch_counters = normalized_counters.get("dispatch")
    dispatch_metrics = metrics.get("dispatch", {})
    if isinstance(dispatch_counters, dict):
        logical_bytes = dispatch_counters.get("logical_outbound_payload_bytes")
        ready_to_all_ms = dispatch_metrics.get("ready_to_all_output_ms")
        if logical_bytes is not None and ready_to_all_ms is not None:
            dispatch_counters = dict(dispatch_counters)
            dispatch_counters["logical_payload_gbps"] = (
                float(logical_bytes) / (ready_to_all_ms * 1e6)
                if ready_to_all_ms > 0
                else None
            )
            normalized_counters["dispatch"] = dispatch_counters

    return {
        "schema": MOE_COMPONENT_PROFILE_SCHEMA,
        "execution_model": execution_model,
        "detail": canonical_detail,
        "events": normalized_events,
        "event_provenance": provenance,
        "capabilities": observed_capabilities,
        "metrics": metrics,
        "counters": normalized_counters,
        "items": normalized_items,
    }


def host_clock_domain_id() -> str:
    """Identify the kernel clock domain backing ``CLOCK_MONOTONIC``."""

    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as source:
            boot_id = source.read().strip()
    except OSError:
        boot_id = "unknown-boot"
    return f"{socket.gethostname()}:{boot_id}"


def calibrated_event_timing_guard_ns() -> int:
    """Read the explicitly calibrated CUDA-event projection guard."""

    raw_value = os.getenv(_EVENT_GUARD_ENV, "").strip()
    if not raw_value:
        raise ValueError(
            f"{_EVENT_GUARD_ENV} must be explicitly set from a matched-host "
            "readiness calibration"
        )
    try:
        guard_ns = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{_EVENT_GUARD_ENV} must be an integer") from error
    if guard_ns < 0:
        raise ValueError(f"{_EVENT_GUARD_ENV} must be nonnegative")
    return guard_ns


def cuda_event_host_interval(
    event: Any,
    anchor: Any,
    *,
    anchor_bracket_start_ns: int,
    anchor_bracket_end_ns: int,
    event_timing_guard_ns: int,
) -> dict[str, int]:
    """Project a CUDA event to a conservative host-monotonic interval.

    The anchor event executes after ``anchor_bracket_start_ns`` and before
    ``anchor_bracket_end_ns``. CUDA elapsed time places ``event`` before that
    unknown anchor instant. ``event_timing_guard_ns`` covers calibrated CUDA
    event quantization and projection residual rather than pretending the
    bracket midpoint is an exact cross-device timestamp.
    """

    if anchor_bracket_end_ns < anchor_bracket_start_ns:
        raise ValueError("CUDA anchor bracket end precedes its start")
    if event_timing_guard_ns < 0:
        raise ValueError("CUDA event timing guard must be nonnegative")
    delta_ns = round(event.elapsed_time(anchor) * 1e6)
    if delta_ns < 0:
        raise ValueError("profiled CUDA event executes after its clock anchor")
    lower_ns = anchor_bracket_start_ns - delta_ns - event_timing_guard_ns
    upper_ns = anchor_bracket_end_ns - delta_ns + event_timing_guard_ns
    return {
        "host_monotonic_ns_estimate": (lower_ns + upper_ns) // 2,
        "host_monotonic_ns_lower": lower_ns,
        "host_monotonic_ns_upper": upper_ns,
    }


def _collector_main(work_queue: queue.Queue[Callable[[], None]]) -> None:
    while True:
        collect = work_queue.get()
        try:
            collect()
        except Exception:
            print("DEEPEP_TIMELINE_COLLECTOR_ERROR", flush=True)
            traceback.print_exc()
        finally:
            work_queue.task_done()


def ensure_moe_timeline_collector() -> None:
    """Start one process-local collector before a selected invocation runs."""

    global _COLLECTOR_QUEUE
    if _COLLECTOR_QUEUE is not None:
        return
    with _COLLECTOR_LOCK:
        if _COLLECTOR_QUEUE is not None:
            return
        queue_size = int(os.getenv("SGLANG_DEEPEP_TIMELINE_QUEUE_SIZE", "4"))
        if queue_size <= 0:
            raise ValueError("SGLANG_DEEPEP_TIMELINE_QUEUE_SIZE must be positive")
        work_queue: queue.Queue[Callable[[], None]] = queue.Queue(
            maxsize=queue_size
        )
        worker = threading.Thread(
            target=_collector_main,
            args=(work_queue,),
            name="sglang-moe-timeline-collector",
            daemon=True,
        )
        worker.start()
        _COLLECTOR_QUEUE = work_queue


def submit_moe_timeline_collection(collect: Callable[[], None]) -> None:
    """Queue collection without synchronizing the serving thread."""

    ensure_moe_timeline_collector()
    assert _COLLECTOR_QUEUE is not None
    try:
        _COLLECTOR_QUEUE.put_nowait(collect)
    except queue.Full as error:
        raise RuntimeError("MoE timeline collector queue is full") from error


@contextmanager
def moe_timeline_scope(state: Optional[dict[str, Any]]) -> Iterator[None]:
    """Expose one selected invocation's event set to nested MoE components."""

    if state is None:
        yield
        return
    token = _ACTIVE_MOE_TIMELINE.set(state)
    try:
        yield
    finally:
        _ACTIVE_MOE_TIMELINE.reset(token)


def get_active_moe_timeline() -> Optional[dict[str, Any]]:
    return _ACTIVE_MOE_TIMELINE.get()


def record_moe_timeline_event(name: str) -> bool:
    """Record ``name`` on the current CUDA stream when a probe is active."""

    state = _ACTIVE_MOE_TIMELINE.get()
    if state is None:
        return False
    event = state["events"].get(name)
    if event is None:
        return False
    event.record()
    state["recorded"].add(name)
    return True


def record_moe_timeline_counter(name: str, value: Any) -> bool:
    """Attach component work metadata to a selected invocation."""

    state = _ACTIVE_MOE_TIMELINE.get()
    if state is None:
        return False
    state.setdefault("counters", {})[name] = value
    return True


def record_moe_timeline_event_after_wait(
    name: str, stream: Any, dependency: Any
) -> bool:
    """Record a selected event on ``stream`` after a device dependency.

    This is used at component boundaries where DeepEP's communication stream,
    rather than the SGLang caller stream, consumes an input.  Waiting through
    the underlying handle avoids firing ``EventOverlap`` post-wait hooks on the
    profiling stream. The wait and timed event are emitted only for a selected
    sample.
    """

    state = _ACTIVE_MOE_TIMELINE.get()
    if state is None:
        return False
    event = state["events"].get(name)
    if event is None:
        return False
    waitable = getattr(dependency, "event", None) or dependency
    if waitable is None or not hasattr(waitable, "current_stream_wait"):
        raise ValueError(f"profile event {name!r} requires a CUDA dependency")

    import torch

    with torch.cuda.stream(stream):
        waitable.current_stream_wait()
        event.record(stream)
    state["recorded"].add(name)
    return True
