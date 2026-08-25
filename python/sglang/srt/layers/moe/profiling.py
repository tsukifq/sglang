"""Strict opt-in, request-local MoE CUDA timeline helpers.

The timeline is carried in a context variable so dispatcher, runner, and
kernel-adapter code can record events without changing production call
signatures.  No event is created unless the outer FusedMoE probe selects the
current layer invocation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import queue
import threading
import traceback
from typing import Any, Callable, Iterator, Optional

_ACTIVE_MOE_TIMELINE: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "sglang_active_moe_timeline", default=None
)
_COLLECTOR_LOCK = threading.Lock()
_COLLECTOR_QUEUE: Optional[queue.Queue[Callable[[], None]]] = None


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
