"""Opt-in staged MoE adapter for the shared component profiler contract."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import torch

from sglang.srt.layers.moe.profiling import (
    build_moe_component_profile,
    calibrated_event_timing_guard_ns,
    cuda_event_host_interval,
    ensure_moe_timeline_collector,
    host_clock_domain_id,
    submit_moe_timeline_collection,
)


_EVENT_NAMES = (
    "dispatch_input_ready",
    "dispatch_prepare_done",
    "dispatch_done",
    "runner_pre_permute_done",
    "w13_start",
    "w13_done",
    "activation_start",
    "activation_done",
    "w2_start",
    "w2_done",
    "runner_post_permute_done",
    "runner_fused_done",
    "gemm_done",
    "combine_prepare_done",
    "combine_done",
    "output_ready",
)


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() not in ("", "0", "false", "no", "n")


class StagedMoeProfiler:
    """Select sparse layer calls and collect their CUDA events off-thread."""

    def __init__(self, *, rank: int, layer_id: int, deepep_enabled: bool) -> None:
        self.rank = rank
        self.layer_id = layer_id
        self.calls = 0
        self.selected_layer = False
        if not (_enabled("SGLANG_DEEPEP_TIMELINE") and deepep_enabled):
            return

        try:
            layer_spec = os.getenv("SGLANG_DEEPEP_TIMELINE_LAYERS", "").strip()
            target_layers = (
                {int(value) for value in layer_spec.split(",") if value.strip()}
                if layer_spec
                else {int(os.getenv("SGLANG_DEEPEP_TIMELINE_LAYER", "0"))}
            )
            self.target_call = int(os.getenv("SGLANG_DEEPEP_TIMELINE_CALL", "0"))
            self.call_stride = int(
                os.getenv("SGLANG_DEEPEP_TIMELINE_CALL_STRIDE", "0")
            )
            self.call_offset = int(
                os.getenv("SGLANG_DEEPEP_TIMELINE_CALL_OFFSET", "0")
            )
        except ValueError as error:
            raise ValueError("MoE timeline layer/call selectors must be integers") from error
        if self.call_stride < 0:
            raise ValueError("SGLANG_DEEPEP_TIMELINE_CALL_STRIDE must be nonnegative")
        if self.call_stride and not 0 <= self.call_offset < self.call_stride:
            raise ValueError("timeline call offset must be within its stride")

        self.detail = os.getenv("SGLANG_DEEPEP_TIMELINE_DETAIL", "full").lower()
        if self.detail not in ("arrival", "full"):
            raise ValueError("SGLANG_DEEPEP_TIMELINE_DETAIL must be arrival or full")
        self.enable_file = os.getenv("SGLANG_DEEPEP_TIMELINE_ENABLE_FILE", "").strip()
        self.run_id = os.getenv("SGLANG_DEEPEP_TIMELINE_RUN_ID", "").strip()
        if not self.run_id:
            raise ValueError("SGLANG_DEEPEP_TIMELINE_RUN_ID is required")
        self.event_guard_ns = calibrated_event_timing_guard_ns()
        self.clock_domain = host_clock_domain_id()
        self.selected_layer = layer_id in target_layers
        if self.selected_layer:
            ensure_moe_timeline_collector()

    def begin(self, hidden_states: torch.Tensor) -> Optional[dict[str, Any]]:
        if not self.selected_layer:
            return None
        call_index = self.calls
        self.calls += 1
        if self.enable_file and not os.path.exists(self.enable_file):
            return None
        if self.call_stride:
            if call_index % self.call_stride != self.call_offset:
                return None
        elif call_index != self.target_call:
            return None

        names = (
            ("dispatch_input_ready", "output_ready")
            if self.detail == "arrival"
            else _EVENT_NAMES
        )
        origin = torch.cuda.Event(enable_timing=True)
        origin.record(torch.cuda.current_stream(hidden_states.device))
        return {
            "context": {
                "rank": self.rank,
                "layer_id": self.layer_id,
                "call_index": call_index,
                "run_id": self.run_id,
                "sample_id": (
                    f"{self.run_id}:staged:layer={self.layer_id}:call={call_index}"
                ),
                "host_clock_domain_id": self.clock_domain,
                "clock_contract": {"event_timing_guard_ns": self.event_guard_ns},
            },
            "origin": origin,
            "timeline": {
                "events": {
                    name: torch.cuda.Event(enable_timing=True) for name in names
                },
                "recorded": set(),
                "counters": {},
            },
            "device": hidden_states.device,
            "input_tokens": hidden_states.size(0),
            "detail": self.detail,
        }

    def submit(self, sample: dict[str, Any]) -> None:
        timeline = sample["timeline"]
        events = dict(timeline["events"])
        recorded = frozenset(timeline["recorded"])
        if "dispatch_input_ready" not in recorded:
            raise ValueError(
                "exact staged readiness requires the DeepEP normal dispatch path"
            )
        if "output_ready" not in recorded:
            raise ValueError("staged MoE profile did not record output_ready")
        counters = dict(timeline.get("counters", {}))
        origin = sample["origin"]
        context = dict(sample["context"])
        output_ready = events["output_ready"]
        device = sample["device"]
        defer_started_ns = time.monotonic_ns()
        defer_state: dict[str, Optional[int]] = {"done_ns": None}

        def collect() -> None:
            collector_started_ns = time.monotonic_ns()
            with torch.cuda.device(device):
                output_ready.synchronize()
                output_wait_done_ns = time.monotonic_ns()
                anchor_stream = torch.cuda.Stream(device=device, priority=0)
                anchor = torch.cuda.Event(enable_timing=True)
                anchor_bracket_start_ns = time.monotonic_ns()
                anchor.record(anchor_stream)
                anchor.synchronize()
                anchor_bracket_end_ns = time.monotonic_ns()

            guard_ns = int(context["clock_contract"]["event_timing_guard_ns"])

            def timestamp(event: torch.cuda.Event) -> dict[str, Any]:
                return {
                    "rank_local_ms": origin.elapsed_time(event),
                    **cuda_event_host_interval(
                        event,
                        anchor,
                        anchor_bracket_start_ns=anchor_bracket_start_ns,
                        anchor_bracket_end_ns=anchor_bracket_end_ns,
                        event_timing_guard_ns=guard_ns,
                    ),
                }

            timestamps = {"layer_entry": timestamp(origin)}
            timestamps.update({name: timestamp(events[name]) for name in sorted(recorded)})
            component_events: dict[str, Any] = {
                "layer_entry": timestamps["layer_entry"],
                "layer_output_ready": timestamps["output_ready"],
            }
            provenance = {
                "layer_entry": "sglang_caller_stream",
                "layer_output_ready": "sglang_caller_stream",
            }
            if "dispatch_input_ready" in recorded:
                component_events["dispatch_input_ready"] = timestamps[
                    "dispatch_input_ready"
                ]
                provenance["dispatch_input_ready"] = (
                    "deepep_comm_stream_after_layout_dependency_before_dispatch_launch"
                )
            if {"dispatch_done", "gemm_done", "combine_done"}.issubset(recorded):
                component_events.update(
                    {
                        "dispatch_first_output_ready": timestamps["dispatch_done"],
                        "dispatch_all_output_ready": timestamps["dispatch_done"],
                        "compute_first_start": timestamps["dispatch_done"],
                        "compute_all_done": timestamps["gemm_done"],
                        "combine_first_start": timestamps["gemm_done"],
                        "combine_all_done": timestamps["combine_done"],
                    }
                )
                provenance.update(
                    {
                        "dispatch_first_output_ready": (
                            "consumer_stream_after_deepep_completion_wait"
                        ),
                        "dispatch_all_output_ready": (
                            "consumer_stream_after_deepep_completion_wait"
                        ),
                        "compute_first_start": "sglang_caller_stream_boundary",
                        "compute_all_done": "sglang_caller_stream_boundary",
                        "combine_first_start": "sglang_caller_stream_boundary",
                        "combine_all_done": (
                            "consumer_stream_after_deepep_completion_wait"
                        ),
                    }
                )

            deep_gemm = counters.get("deep_gemm")
            if isinstance(deep_gemm, dict) and {
                "w13_start",
                "w13_done",
                "w2_start",
                "w2_done",
            }.issubset(recorded):
                deep_gemm = dict(deep_gemm)
                w13_ms = events["w13_start"].elapsed_time(events["w13_done"])
                w2_ms = events["w2_start"].elapsed_time(events["w2_done"])
                gemm_ms = w13_ms + w2_ms
                deep_gemm["measured"] = {
                    "w13_ms": w13_ms,
                    "w13_useful_tflops": (
                        deep_gemm["w13"]["useful_flops"] / (w13_ms * 1e9)
                        if w13_ms > 0
                        else None
                    ),
                    "w2_ms": w2_ms,
                    "w2_useful_tflops": (
                        deep_gemm["w2"]["useful_flops"] / (w2_ms * 1e9)
                        if w2_ms > 0
                        else None
                    ),
                    "gemm_ms": gemm_ms,
                    "gemm_useful_tflops": (
                        deep_gemm["useful_flops"] / (gemm_ms * 1e9)
                        if gemm_ms > 0
                        else None
                    ),
                }

            component_profile = build_moe_component_profile(
                execution_model="staged",
                detail=sample["detail"],
                events=component_events,
                event_provenance=provenance,
                counters={
                    "input_tokens": sample["input_tokens"],
                    "compute": {"deep_gemm": deep_gemm},
                },
                capabilities={
                    "exact_dispatch_input_ready": "dispatch_input_ready" in recorded,
                    "exact_dispatch_output_ready": "dispatch_done" in recorded,
                    "physical_bytes": False,
                    "hardware_counters": False,
                },
            )
            collector_before_log_ns = time.monotonic_ns()
            serving_done_ns = defer_state["done_ns"]
            payload = {
                "schema": "sglang-deepep-staged-timeline-v1",
                **context,
                "input_tokens": sample["input_tokens"],
                "profiler_overhead": {
                    "serving_thread_synchronized": False,
                    "serving_thread_defer_us": (
                        (serving_done_ns - defer_started_ns) / 1e3
                        if serving_done_ns is not None
                        else None
                    ),
                    "collector_queue_delay_us": (
                        collector_started_ns - defer_started_ns
                    )
                    / 1e3,
                    "collector_wait_for_output_ms": (
                        output_wait_done_ns - collector_started_ns
                    )
                    / 1e6,
                    "collector_before_log_ms": (
                        collector_before_log_ns - collector_started_ns
                    )
                    / 1e6,
                    "metadata_d2h_bytes": 0,
                    "timed_cuda_event_count": len(recorded) + 1,
                },
                "clock_alignment": {
                    "method": "bracketed private-stream CUDA anchor",
                    "anchor_bracket_start_ns": anchor_bracket_start_ns,
                    "anchor_bracket_end_ns": anchor_bracket_end_ns,
                    "event_timing_guard_ns": guard_ns,
                },
                "component_profile": component_profile,
            }
            print("DEEPEP_STAGED_TIMELINE " + json.dumps(payload), flush=True)

        submit_moe_timeline_collection(collect)
        defer_state["done_ns"] = time.monotonic_ns()
