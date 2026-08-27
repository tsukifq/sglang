from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
import types

import pytest


MOE_ROOT = Path(__file__).parents[1] / "srt" / "layers" / "moe"


def _load_profiler_modules():
    for package in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.moe"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module

    profiling_name = "sglang.srt.layers.moe.profiling"
    profiling_spec = importlib.util.spec_from_file_location(
        profiling_name, MOE_ROOT / "profiling.py"
    )
    assert profiling_spec is not None and profiling_spec.loader is not None
    profiling = importlib.util.module_from_spec(profiling_spec)
    sys.modules[profiling_name] = profiling
    profiling_spec.loader.exec_module(profiling)

    staged_name = "sglang.srt.layers.moe.staged_profiling"
    staged_spec = importlib.util.spec_from_file_location(
        staged_name, MOE_ROOT / "staged_profiling.py"
    )
    assert staged_spec is not None and staged_spec.loader is not None
    staged = importlib.util.module_from_spec(staged_spec)
    sys.modules[staged_name] = staged
    staged_spec.loader.exec_module(staged)
    return profiling, staged


PROFILING, STAGED = _load_profiler_modules()


def test_disabled_staged_profiler_does_not_select_a_layer(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_DEEPEP_TIMELINE", raising=False)

    profiler = STAGED.StagedMoeProfiler(rank=0, layer_id=1, deepep_enabled=True)

    assert profiler.selected_layer is False


def test_enabled_staged_profiler_fails_closed_without_run_identity(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SGLANG_DEEPEP_TIMELINE", "1")
    monkeypatch.setenv("SGLANG_DEEPEP_TIMELINE_LAYER", "1")
    monkeypatch.setenv("SGLANG_DEEPEP_TIMELINE_EVENT_GUARD_NS", "1000")
    monkeypatch.delenv("SGLANG_DEEPEP_TIMELINE_RUN_ID", raising=False)

    with pytest.raises(ValueError, match="RUN_ID is required"):
        STAGED.StagedMoeProfiler(rank=0, layer_id=1, deepep_enabled=True)


def test_staged_profiler_rejects_missing_exact_readiness_event() -> None:
    profiler = object.__new__(STAGED.StagedMoeProfiler)
    sample = {
        "timeline": {
            "events": {"output_ready": object()},
            "recorded": {"output_ready"},
        }
    }

    with pytest.raises(ValueError, match="requires the DeepEP normal dispatch path"):
        profiler.submit(sample)


def test_staged_component_contract_collapses_dispatch_readiness() -> None:
    profile = PROFILING.build_moe_component_profile(
        execution_model="staged",
        detail="full",
        events={
            "layer_entry": 0.0,
            "dispatch_input_ready": 0.1,
            "dispatch_first_output_ready": 0.4,
            "dispatch_all_output_ready": 0.4,
            "compute_first_start": 0.4,
            "compute_all_done": 0.8,
            "combine_first_start": 0.8,
            "combine_all_done": 1.0,
            "layer_output_ready": 1.0,
        },
    )

    assert profile["execution_model"] == "staged"
    assert profile["metrics"]["dispatch"]["ready_to_all_output_ms"] == pytest.approx(
        0.3
    )
    assert profile["metrics"]["dispatch"]["output_readiness_spread_ms"] == 0.0


def test_dispatch_input_ready_hook_precedes_deepep_dispatch_launch() -> None:
    source = (
        MOE_ROOT
        / "token_dispatcher"
        / "deepep.py"
    ).read_text(encoding="utf-8")
    normal = source.index("class _DeepEPDispatcherImplNormal")
    core_start = source.index("    def _dispatch_core(", normal)
    core = source[core_start : source.index("    def combine_a(", core_start)]

    layout_done = core.index("buffer.get_dispatch_layout(")
    readiness = core.index("_record_dispatch_input_ready(buffer, previous_event)")
    dispatch_launch = core.index("buffer.dispatch(")
    assert layout_done < readiness < dispatch_launch
    assert "buffer.get_comm_stream()" in source


def test_dispatch_input_ready_waits_on_raw_event_without_overlap_hook(
    monkeypatch,
) -> None:
    class FakeWaitable:
        waits = 0

        def current_stream_wait(self) -> None:
            self.waits += 1

    class FakeDependency:
        def __init__(self, event) -> None:
            self.event = event

        def current_stream_wait(self) -> None:
            raise AssertionError("EventOverlap hook path must not be used")

    class FakeEvent:
        def __init__(self) -> None:
            self.streams = []

        def record(self, stream) -> None:
            self.streams.append(stream)

    monkeypatch.setattr(STAGED.torch.cuda, "stream", lambda stream: nullcontext())
    waitable = FakeWaitable()
    dependency = FakeDependency(waitable)
    stream = object()
    event = FakeEvent()
    state = {"events": {"dispatch_input_ready": event}, "recorded": set()}

    with PROFILING.moe_timeline_scope(state):
        assert PROFILING.record_moe_timeline_event_after_wait(
            "dispatch_input_ready", stream, dependency
        )

    assert waitable.waits == 1
    assert event.streams == [stream]
    assert state["recorded"] == {"dispatch_input_ready"}
