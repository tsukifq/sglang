"""Inference-only lane-streaming DeepEP consumer.

This module is intentionally separate from the regular dispatcher/runner
formats.  The experimental path consumes the lane-local sidecar exported by
Async MoE's DeepEP fork, runs a complete BF16 or block-FP8 expert MLP per
independently ready source lane, and uses source-local streaming combine
kernels.  The regular DeepEP path remains the default.
"""

from __future__ import annotations

import json
import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from sglang.srt.layers.moe.deepep_streaming_kernels import (
    build_rank_merged_expert_layout,
    masked_route_weight_mul_,
    pack_rank_merged_rows,
    scatter_rank_merged_rows,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.profiling import (
    best_cuda_clock_anchor,
    build_moe_component_profile,
    canonical_moe_profile_detail,
    cuda_event_host_interval,
    submit_moe_timeline_collection,
)

_DEEPEP_STREAMING_REQUIRED_ENV = {
    "EP_EXPERIMENTAL_STREAMING_LANES": "1",
    "EP_EXPERIMENTAL_STREAMING_LAYER": "1",
    # NCCL Device API windows require the cuMem allocator. SGLang otherwise
    # disables it unless its separate symmetric-memory feature is selected.
    "NCCL_CUMEM_ENABLE": "1",
    # PyTorch communicators are not guaranteed to be created with the NCCL
    # Device API symmetric-memory capability required by ElasticBuffer.
    # Create and cache one DeepEP-managed communicator for the EP group.
    "EP_REUSE_NCCL_COMM": "0",
}
_DEEPEP_STREAMING_INCOMPATIBLE_ENV = (
    "EP_EXPERIMENTAL_STREAMING_COPY_SHADOW",
    "EP_EXPERIMENTAL_RANK_READY",
    "EP_EXPERIMENTAL_STREAMING_INSTRUMENTED_BULK",
)


def is_deepep_streaming_enabled() -> bool:
    return envs.SGLANG_ENABLE_DEEPEP_STREAMING.get()


def is_deepep_v2_sync_baseline_enabled() -> bool:
    """Gate all V2 source lanes on the slowest arrival for fair baselines."""

    return os.getenv("SGLANG_DEEPEP_V2_SYNC_BASELINE", "0").lower() not in (
        "",
        "0",
        "false",
        "no",
        "n",
    )


def is_deepep_streaming_rank_merge_enabled() -> bool:
    """Coalesce equal experts across ready source lanes on the owner rank."""

    return os.getenv(
        "SGLANG_DEEPEP_STREAMING_RANK_MERGE", "0"
    ).lower() not in (
        "",
        "0",
        "false",
        "no",
        "n",
    )


def configure_deepep_streaming_environment() -> None:
    """Enable the matching DeepEP protocol and reject mixed prototypes."""

    for name in _DEEPEP_STREAMING_INCOMPATIBLE_ENV:
        if os.getenv(name, "0").lower() not in ("", "0", "false", "no", "n"):
            raise RuntimeError(
                f"SGLANG_ENABLE_DEEPEP_STREAMING is incompatible with {name}"
            )
    for name, required in _DEEPEP_STREAMING_REQUIRED_ENV.items():
        current = os.getenv(name)
        if current is not None and current != required:
            raise RuntimeError(
                f"SGLANG_ENABLE_DEEPEP_STREAMING requires {name}={required}, "
                f"got {current!r}"
            )
        os.environ[name] = required


def _resolve_streaming_wave_size(
    *, lanes: int, device_major: int, grouped_gemm: Callable[..., Any]
) -> int:
    """Choose a wave size supported by the loaded DeepGEMM extension."""

    try:
        supports_repeated_weights = (
            "repeat_weight_groups" in inspect.signature(grouped_gemm).parameters
        )
    except (TypeError, ValueError):
        # pybind11 callables commonly omit __text_signature__, so
        # inspect.signature raises even though their generated docstring has
        # the complete C++ binding declaration.
        declarations = (
            getattr(grouped_gemm, "__text_signature__", None),
            getattr(grouped_gemm, "__doc__", None),
        )
        supports_repeated_weights = any(
            declaration
            and "repeat_weight_groups" in declaration.splitlines()[0]
            for declaration in declarations
        )

    configured = os.getenv("SGLANG_DEEPEP_STREAMING_WAVE_SIZE")
    if configured is None:
        return (
            4
            if device_major >= 10
            and lanes % 4 == 0
            and supports_repeated_weights
            else 1
        )

    wave_size = int(configured)
    if wave_size <= 0:
        raise ValueError("SGLANG_DEEPEP_STREAMING_WAVE_SIZE must be positive")
    if wave_size > 1 and not supports_repeated_weights:
        raise RuntimeError(
            "the loaded DeepGEMM extension does not support "
            "repeat_weight_groups; rebuild DeepGEMM or set "
            "SGLANG_DEEPEP_STREAMING_WAVE_SIZE=1"
        )
    return wave_size


@dataclass(frozen=True, slots=True)
class DeepEPStreamingDispatch:
    """One generation of lane-local dispatch state."""

    buffer: Any
    x: torch.Tensor
    sf: torch.Tensor | None
    route_weights: torch.Tensor | None
    src_metadata: torch.Tensor
    expert_psum: torch.Tensor
    pack_done_seq: torch.Tensor
    generation: int
    source_topk_idx: torch.Tensor
    transport_handle: Any
    transport_event: Any
    profile_events: dict[str, torch.cuda.Event] | None = None
    wavefront_slot: int = 0

    @classmethod
    def from_runtime(
        cls,
        *,
        buffer: Any,
        raw: tuple[Any, ...],
        source_topk_idx: torch.Tensor,
        transport_handle: Any,
        transport_event: Any,
        profile_events: dict[str, torch.cuda.Event] | None = None,
        wavefront_slot: int = 0,
    ) -> "DeepEPStreamingDispatch":
        if len(raw) != 7:
            raise ValueError(f"expected seven DeepEP lane-view fields, got {len(raw)}")
        view = cls(
            buffer,
            *raw,
            source_topk_idx,
            transport_handle,
            transport_event,
            dict(profile_events or {}),
            wavefront_slot,
        )
        view.validate()
        return view

    def validate(self) -> None:
        if self.generation <= 0:
            raise ValueError("DeepEP streaming generation must be positive")
        if self.wavefront_slot < 0:
            raise ValueError("DeepEP streaming wavefront slot must be non-negative")
        if self.x.ndim != 3:
            raise ValueError(
                "lane activation must have shape [lanes, capacity, hidden]"
            )
        lanes = self.x.size(0)
        if self.x.dtype == torch.bfloat16:
            if self.sf is not None:
                raise ValueError("BF16 streaming payload must not carry FP8 scales")
        elif self.x.dtype == torch.float8_e4m3fn:
            if self.sf is None or self.sf.ndim != 3:
                raise ValueError(
                    "FP8 streaming payload requires lane-local activation scales"
                )
            if tuple(self.sf.shape[:2]) != tuple(self.x.shape[:2]):
                raise ValueError(
                    "FP8 activation scales must match payload lanes and capacity"
                )
        else:
            raise ValueError(f"unsupported streaming payload dtype {self.x.dtype}")
        if self.expert_psum.ndim != 2 or self.expert_psum.size(0) != lanes:
            raise ValueError("expert psum must have shape [lanes, local_experts]")
        if self.pack_done_seq.ndim != 1 or self.pack_done_seq.size(0) != lanes:
            raise ValueError("pack doorbells must have shape [lanes]")
        if self.route_weights is None:
            raise ValueError("streaming MoE requires routed top-k weights")
        if tuple(self.route_weights.shape[:2]) != tuple(self.x.shape[:2]):
            raise ValueError("route weights must have shape [lanes, capacity]")
        if self.src_metadata.ndim != 2 or self.src_metadata.size(0) % lanes != 0:
            raise ValueError("source metadata must be evenly partitioned by lane")
        if self.source_topk_idx.ndim != 2 or not self.source_topk_idx.is_contiguous():
            raise ValueError("source top-k ids must be a contiguous rank-local matrix")
        if self.src_metadata.size(1) != self.source_topk_idx.size(1) + 2:
            raise ValueError("source metadata and top-k widths disagree")
        for tensor in (
            self.x,
            self.sf,
            self.route_weights,
            self.src_metadata,
            self.expert_psum,
            self.pack_done_seq,
            self.source_topk_idx,
        ):
            if tensor is not None and not tensor.is_cuda:
                raise ValueError("DeepEP streaming tensors must be CUDA-resident")


@dataclass(frozen=True, slots=True)
class DeepEPStreamingLayerResult:
    """Source-local output plus objects that protect the in-flight epoch."""

    output: torch.Tensor
    source_ready: Any
    source_stream: torch.cuda.Stream
    epoch_drained: torch.cuda.Event
    lane_return_done: tuple[torch.cuda.Event, ...]
    dispatch: DeepEPStreamingDispatch

    def handoff_to_stream(self, stream: torch.cuda.Stream) -> torch.Tensor:
        """Add only the local source dependency when changing CUDA streams."""

        if stream.device != self.source_stream.device:
            raise ValueError("streaming output cannot cross CUDA devices")
        if stream.cuda_stream != self.source_stream.cuda_stream:
            with torch.cuda.stream(stream):
                self.source_ready.current_stream_wait()
        self.output.record_stream(stream)
        return self.output


def _require_per_lane_release(buffer: Any) -> Callable[[int, int], None]:
    """Return the v2 generation-aware lane releaser or fail before submission.

    Protocol v1 publishes its safe-release marker in combine-return, so
    queuing that ACK before GEMM would deadlock the lane stream. Falling back
    to destination-wide ``release_streaming_lane_view`` after all returns
    would restore the slowest-lane coupling. Protocol v2 snapshots every
    downstream input during pack and can acknowledge ingress before GEMM.
    """

    release_lane = getattr(buffer, "release_streaming_lane", None)
    finalize_view = getattr(buffer, "release_streaming_lane_view", None)
    protocol_version = getattr(buffer, "get_streaming_lane_protocol_version", None)
    runtime = getattr(buffer, "runtime", None)
    runtime_release_lane = (
        getattr(runtime, "release_streaming_lane", None)
        if runtime is not None
        else release_lane
    )
    runtime_protocol_version = (
        getattr(runtime, "get_streaming_lane_protocol_version", None)
        if runtime is not None
        else protocol_version
    )
    if (
        not callable(release_lane)
        or not callable(finalize_view)
        or not callable(runtime_release_lane)
        or not callable(protocol_version)
        or not callable(runtime_protocol_version)
    ):
        raise RuntimeError(
            "streaming DeepEP requires protocol v2 and the generation-aware "
            "buffer.release_streaming_lane(source_rank, generation) API; "
            "refusing the destination-wide release_streaming_lane_view "
            "fallback because it is not per-lane safe"
        )
    wrapper_version = protocol_version()
    native_version = runtime_protocol_version()
    if wrapper_version != 2 or native_version != 2:
        raise RuntimeError(
            "streaming DeepEP requires lane protocol v2 before transport; "
            f"wrapper={wrapper_version!r}, native={native_version!r}"
        )
    return release_lane


def _check_cuda_driver(result: tuple[Any, ...], operation: str, cuda: Any) -> None:
    if result != (cuda.CUresult.CUDA_SUCCESS,):
        raise RuntimeError(f"{operation} failed: {result}")


def _lane_layout_from_psum(psum: Sequence[int]) -> dict[str, Any]:
    """Decode useful expert rows and alignment holes from DeepGEMM psums."""

    starts = []
    counts = []
    previous_end = 0
    for end in psum:
        start = (previous_end + 127) // 128 * 128
        starts.append(start)
        counts.append(end - start)
        previous_end = end
    useful_rows = sum(counts)
    active_span_rows = psum[-1] if psum else 0
    return {
        "useful_rows": useful_rows,
        "active_span_rows": active_span_rows,
        "alignment_hole_rows": active_span_rows - useful_rows,
        "nonempty_experts": sum(count > 0 for count in counts),
        "max_expert_rows": max(counts, default=0),
        "expert_rows": counts,
        "expert_starts": starts,
    }


def _logical_outbound_dispatch_from_host(
    topk_idx: torch.Tensor,
    lanes: int,
    local_experts: int,
    hidden_bytes: int,
    route_metadata_bytes: int,
) -> list[dict[str, int]]:
    """Count routed payload bytes without claiming physical link traffic.

    DeepEP sends one activation to a destination when any of the token's
    routes lands there, plus the token's complete top-k ids and weights. The
    count excludes protocol headers, count matrices, and cache-line traffic.
    """

    destinations = []
    for destination in range(lanes):
        lower = destination * local_experts
        upper = lower + local_experts
        routed = (topk_idx >= lower) & (topk_idx < upper)
        unique_tokens = int(routed.any(dim=1).sum().item())
        routes = int(routed.sum().item())
        destinations.append(
            {
                "destination_rank": destination,
                "unique_tokens": unique_tokens,
                "routes": routes,
                "logical_payload_bytes": unique_tokens
                * (hidden_bytes + route_metadata_bytes),
            }
        )
    return destinations


def _emit_streaming_timeline(
    dispatch: DeepEPStreamingDispatch,
    context: dict[str, Any],
    origin: torch.cuda.Event,
    dispatch_done: torch.cuda.Event,
    lane_events: Sequence[dict[str, Any]],
    reduce_start: torch.cuda.Event,
    reduce_done: torch.cuda.Event,
) -> None:
    """Queue one streaming sample for collection off the serving thread."""

    defer_started_ns = time.monotonic_ns()
    device = dispatch.x.device
    lane_events = tuple(dict(events) for events in lane_events)
    # The lane view owns this generation's packed control tensor, so later
    # generations may reuse the symmetric ingress workspace without changing
    # the psums consumed by this deferred collector.
    expert_psum = dispatch.expert_psum.detach()
    source_topk_idx = dispatch.source_topk_idx.detach()
    generation = dispatch.generation
    input_tokens = source_topk_idx.size(0)
    lane_count = dispatch.x.size(0)
    local_experts = expert_psum.size(1)
    lane_capacity_rows = dispatch.x.size(1)
    hidden_bytes = dispatch.x.size(2) * dispatch.x.element_size()
    route_metadata_bytes = source_topk_idx.size(1) * (
        source_topk_idx.element_size()
        + torch.tensor([], dtype=torch.float32).element_size()
    )
    context = dict(context)
    defer_state = {"done_ns": None}
    arrival_only = context["profile_detail"] == "arrival"

    def collect() -> None:
        collector_started_ns = time.monotonic_ns()
        with torch.cuda.device(device):
            reduce_done.synchronize()
            if not arrival_only:
                dispatch_done.synchronize()
                for events in lane_events:
                    events["return_done"].synchronize()
            output_wait_done_ns = time.monotonic_ns()

            anchor_selection = best_cuda_clock_anchor(torch.cuda, device)
            profile_stream = anchor_selection["stream"]
            clock_anchor = anchor_selection["event"]
            anchor_bracket_start_ns = anchor_selection["bracket_start_ns"]
            anchor_bracket_end_ns = anchor_selection["bracket_end_ns"]

            psum_host = None
            topk_host = None
            if not arrival_only:
                psum_host = torch.empty(
                    expert_psum.shape,
                    dtype=expert_psum.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                topk_host = torch.empty(
                    source_topk_idx.shape,
                    dtype=source_topk_idx.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                metadata_done = torch.cuda.Event()
                with torch.cuda.stream(profile_stream):
                    psum_host.copy_(expert_psum, non_blocking=True)
                    topk_host.copy_(source_topk_idx, non_blocking=True)
                    metadata_done.record(profile_stream)
                metadata_done.synchronize()

        anchor_midpoint_ns = (anchor_bracket_start_ns + anchor_bracket_end_ns) // 2

        def aligned_timestamp(event: torch.cuda.Event) -> dict[str, float | int]:
            rank_local_ms = origin.elapsed_time(event)
            return {
                "rank_local_ms": rank_local_ms,
                **cuda_event_host_interval(
                    event,
                    clock_anchor,
                    anchor_bracket_start_ns=anchor_bracket_start_ns,
                    anchor_bracket_end_ns=anchor_bracket_end_ns,
                    event_timing_guard_ns=int(
                        context["clock_contract"]["event_timing_guard_ns"]
                    ),
                ),
            }

        if arrival_only:
            layer_entry = aligned_timestamp(origin)
            output_ready = aligned_timestamp(reduce_done)
            component_profile = build_moe_component_profile(
                execution_model="streaming",
                detail=context["profile_detail"],
                events={
                    "layer_entry": layer_entry,
                    "layer_output_ready": output_ready,
                },
                event_provenance={
                    "layer_entry": (
                        "sglang_caller_stream_event_recorded_at_moe_python_entry"
                    ),
                    "layer_output_ready": "source_reduce_stream",
                },
                counters={"input_tokens": input_tokens},
                capabilities={
                    "layer_entry_includes_host_launch_arrival": True,
                    "exact_dispatch_output_ready": False,
                },
            )
            collector_before_log_ns = time.monotonic_ns()
            serving_done_ns = defer_state["done_ns"]
            payload = {
                "schema": "sglang-deepep-streaming-timeline-v3",
                **context,
                "generation": generation,
                "input_tokens": input_tokens,
                "lane_capacity_rows": lane_capacity_rows,
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
                    "timed_cuda_event_count": 2,
                },
                "clock_alignment": {
                    "method": (
                        "minimum-width repeated private-stream CUDA anchor "
                        "projected to host CLOCK_MONOTONIC"
                    ),
                    "anchor_host_monotonic_ns_midpoint": anchor_midpoint_ns,
                    "anchor_bracket_start_ns": anchor_bracket_start_ns,
                    "anchor_bracket_end_ns": anchor_bracket_end_ns,
                    "anchor_attempts": anchor_selection["attempts"],
                    "anchor_selected_attempt": anchor_selection[
                        "selected_attempt"
                    ],
                    "anchor_bracket_widths_ns": anchor_selection[
                        "bracket_widths_ns"
                    ],
                    "uncertainty_ns": (
                        anchor_bracket_end_ns - anchor_bracket_start_ns + 1
                    )
                    // 2
                    + int(context["clock_contract"]["event_timing_guard_ns"]),
                    "event_timing_guard_ns": int(
                        context["clock_contract"]["event_timing_guard_ns"]
                    ),
                },
                "arrival_timestamps": {
                    "moe_entry": layer_entry,
                    "output_ready": output_ready,
                },
                "component_profile": component_profile,
            }
            print("DEEPEP_STREAMING_TIMELINE " + json.dumps(payload), flush=True)
            return

        assert psum_host is not None
        assert topk_host is not None

        psums = psum_host.to(dtype=torch.int64).tolist()
        outbound = _logical_outbound_dispatch_from_host(
            topk_host,
            lane_count,
            local_experts,
            hidden_bytes,
            route_metadata_bytes,
        )
        dispatch_ms = origin.elapsed_time(dispatch_done)
        total_logical_bytes = sum(item["logical_payload_bytes"] for item in outbound)
        lanes = []
        lane_arrivals = []
        for source_rank, (events, psum) in enumerate(zip(lane_events, psums)):
            ready_ms = origin.elapsed_time(events["gemm_start"])
            gemm_done_ms = origin.elapsed_time(events["gemm_done"])
            return_start_ms = origin.elapsed_time(events["return_start"])
            return_done_ms = origin.elapsed_time(events["return_done"])
            lanes.append(
                {
                    "source_rank": source_rank,
                    "compute_scope": events.get("compute_scope", "lane"),
                    "compute_group_id": events.get(
                        "compute_group_id", source_rank
                    ),
                    **_lane_layout_from_psum(psum),
                    "gemm_start_ms": ready_ms,
                    "gemm_done_ms": gemm_done_ms,
                    "gemm_ms": events["gemm_start"].elapsed_time(events["gemm_done"]),
                    "return_start_ms": return_start_ms,
                    "return_done_ms": return_done_ms,
                    "return_ms": events["return_start"].elapsed_time(
                        events["return_done"]
                    ),
                }
            )
            lane_arrivals.append(
                {
                    "source_rank": source_rank,
                    "gemm_start": aligned_timestamp(events["gemm_start"]),
                    "gemm_done": aligned_timestamp(events["gemm_done"]),
                    "return_start": aligned_timestamp(events["return_start"]),
                    "return_done": aligned_timestamp(events["return_done"]),
                }
            )
        first_compute_lane = min(
            range(len(lanes)), key=lambda lane: lanes[lane]["gemm_start_ms"]
        )
        last_compute_lane = max(
            range(len(lanes)), key=lambda lane: lanes[lane]["gemm_done_ms"]
        )
        first_combine_lane = min(
            range(len(lanes)), key=lambda lane: lanes[lane]["return_start_ms"]
        )
        first_consumer_lane = min(
            range(len(lanes)), key=lambda lane: lanes[lane]["gemm_start_ms"]
        )
        last_consumer_lane = max(
            range(len(lanes)), key=lambda lane: lanes[lane]["gemm_start_ms"]
        )
        layer_entry = aligned_timestamp(origin)
        dispatch_transport_done = aligned_timestamp(dispatch_done)
        combine_all_done = aligned_timestamp(reduce_done)
        profile_events = dispatch.profile_events or {}
        dispatch_input_ready = (
            aligned_timestamp(profile_events["dispatch_input_ready"])
            if "dispatch_input_ready" in profile_events
            else None
        )
        canonical_detail = canonical_moe_profile_detail(
            context["profile_detail"], "streaming"
        )
        component_items = []
        if canonical_detail in ("lane", "hardware"):
            component_items = [
                {
                    "kind": "source_lane",
                    "id": str(lane["source_rank"]),
                    "events": {
                        "dispatch_consumer_start": arrival["gemm_start"],
                        "compute_start": arrival["gemm_start"],
                        "compute_done": arrival["gemm_done"],
                        "combine_start": arrival["return_start"],
                        "combine_done": arrival["return_done"],
                    },
                    "counters": {
                        "compute_scope": lane["compute_scope"],
                        "compute_group_id": lane["compute_group_id"],
                        "useful_rows": lane["useful_rows"],
                        "active_span_rows": lane["active_span_rows"],
                        "alignment_hole_rows": lane["alignment_hole_rows"],
                        "nonempty_experts": lane["nonempty_experts"],
                    },
                }
                for lane, arrival in zip(lanes, lane_arrivals)
            ]
        component_events = {
            "layer_entry": layer_entry,
            "dispatch_transport_done": dispatch_transport_done,
            # A lane event is recorded after its device doorbell wait and
            # immediately before GEMM. It is a consumer-start observation,
            # not an exact DeepEP publication timestamp.
            "dispatch_first_consumer_start": lane_arrivals[first_consumer_lane][
                "gemm_start"
            ],
            "dispatch_all_consumer_start": lane_arrivals[last_consumer_lane][
                "gemm_start"
            ],
            "compute_first_start": lane_arrivals[first_compute_lane]["gemm_start"],
            "compute_all_done": lane_arrivals[last_compute_lane]["gemm_done"],
            "combine_first_start": lane_arrivals[first_combine_lane][
                "return_start"
            ],
            "combine_all_done": combine_all_done,
            "layer_output_ready": combine_all_done,
        }
        component_provenance = {
            "layer_entry": (
                "sglang_caller_stream_event_recorded_at_moe_python_entry"
            ),
            "dispatch_transport_done": "deepep_transport_completion_event",
            "dispatch_first_consumer_start": (
                "lane_stream_after_deepep_doorbell_wait_proxy"
            ),
            "dispatch_all_consumer_start": (
                "lane_stream_after_deepep_doorbell_wait_proxy"
            ),
            "compute_first_start": "lane_compute_stream",
            "compute_all_done": "lane_compute_stream",
            "combine_first_start": "lane_return_stream",
            "combine_all_done": "source_reduce_stream",
            "layer_output_ready": "source_reduce_stream",
        }
        if dispatch_input_ready is not None:
            component_events["dispatch_input_ready"] = dispatch_input_ready
            component_provenance["dispatch_input_ready"] = (
                "deepep_comm_stream_after_input_dependency_wait"
            )
        component_profile = build_moe_component_profile(
            execution_model="streaming",
            detail=context["profile_detail"],
            events=component_events,
            event_provenance=component_provenance,
            counters={
                "input_tokens": input_tokens,
                "dispatch": {
                    "logical_outbound_payload_bytes": total_logical_bytes,
                    "lane_capacity_rows": lane_capacity_rows,
                },
                "compute": {
                    "useful_rows": sum(lane["useful_rows"] for lane in lanes),
                    "active_span_rows": sum(
                        lane["active_span_rows"] for lane in lanes
                    ),
                    "nonempty_experts": sum(
                        lane["nonempty_experts"] for lane in lanes
                    ),
                },
            },
            items=component_items,
            capabilities={
                "layer_entry_includes_host_launch_arrival": True,
                "exact_dispatch_input_ready": dispatch_input_ready is not None,
                "exact_dispatch_output_ready": False,
                "exact_per_lane_compute": all(
                    lane["compute_scope"] == "lane" for lane in lanes
                ),
            },
        )
        collector_before_log_ns = time.monotonic_ns()
        serving_done_ns = defer_state["done_ns"]
        payload = {
            "schema": "sglang-deepep-streaming-timeline-v3",
            **context,
            "generation": generation,
            "input_tokens": input_tokens,
            "lane_capacity_rows": lane_capacity_rows,
            "profiler_overhead": {
                "serving_thread_synchronized": False,
                "serving_thread_defer_us": (
                    (serving_done_ns - defer_started_ns) / 1e3
                    if serving_done_ns is not None
                    else None
                ),
                "collector_queue_delay_us": (collector_started_ns - defer_started_ns)
                / 1e3,
                "collector_wait_for_output_ms": (
                    output_wait_done_ns - collector_started_ns
                )
                / 1e6,
                "collector_before_log_ms": (
                    collector_before_log_ns - collector_started_ns
                )
                / 1e6,
                "metadata_d2h_bytes": (
                    psum_host.numel() * psum_host.element_size()
                    + topk_host.numel() * topk_host.element_size()
                ),
                "timed_cuda_event_count": len(
                    {
                        id(events[name])
                        for events in lane_events
                        for name in (
                            "gemm_start",
                            "gemm_done",
                            "return_start",
                            "return_done",
                        )
                    }
                )
                + 3
                + len(profile_events),
            },
            "clock_alignment": {
                "method": (
                    "minimum-width repeated private-stream CUDA anchor "
                    "projected to host CLOCK_MONOTONIC"
                ),
                "anchor_host_monotonic_ns_midpoint": anchor_midpoint_ns,
                "anchor_bracket_start_ns": anchor_bracket_start_ns,
                "anchor_bracket_end_ns": anchor_bracket_end_ns,
                "anchor_attempts": anchor_selection["attempts"],
                "anchor_selected_attempt": anchor_selection[
                    "selected_attempt"
                ],
                "anchor_bracket_widths_ns": anchor_selection[
                    "bracket_widths_ns"
                ],
                "uncertainty_ns": (
                    anchor_bracket_end_ns - anchor_bracket_start_ns + 1
                )
                // 2
                + int(context["clock_contract"]["event_timing_guard_ns"]),
                "event_timing_guard_ns": int(
                    context["clock_contract"]["event_timing_guard_ns"]
                ),
            },
            "arrival_timestamps": {
                "moe_entry": layer_entry,
                **(
                    {"dispatch_input_ready": dispatch_input_ready}
                    if dispatch_input_ready is not None
                    else {}
                ),
                "dispatch_transport_done": dispatch_transport_done,
                "combine_reduce_start": aligned_timestamp(reduce_start),
                "combine_reduce_done": combine_all_done,
                "lanes": lane_arrivals,
            },
            "dispatch": {
                "transport_event_done_ms": dispatch_ms,
                "logical_outbound_payload_bytes": total_logical_bytes,
                "logical_outbound_gbps_at_transport_event": total_logical_bytes
                / (dispatch_ms * 1e6),
                "first_lane_consumer_visible_ms": min(
                    lane["gemm_start_ms"] for lane in lanes
                ),
                "all_lanes_consumer_visible_ms": max(
                    lane["gemm_start_ms"] for lane in lanes
                ),
                "destinations": outbound,
            },
            "lanes": lanes,
            "combine_reduce": {
                "start_ms": origin.elapsed_time(reduce_start),
                "done_ms": origin.elapsed_time(reduce_done),
                "elapsed_ms_including_return_wait": reduce_start.elapsed_time(
                    reduce_done
                ),
            },
            "component_profile": component_profile,
        }
        print("DEEPEP_STREAMING_TIMELINE " + json.dumps(payload), flush=True)

    submit_moe_timeline_collection(collect)
    defer_state["done_ns"] = time.monotonic_ns()


def _launch_streaming_moe_lanes(
    dispatch: DeepEPStreamingDispatch,
    lane_compute: Callable[[int, torch.Tensor], Sequence[torch.Tensor]],
    persistent_tensors: Sequence[torch.Tensor],
    *,
    streams: Sequence[torch.cuda.Stream] | None,
    drain_stream: torch.cuda.Stream | None,
    timeline_context: dict[str, Any] | None,
    timeline_origin: torch.cuda.Event | None,
) -> DeepEPStreamingLayerResult:
    """Submit one expert runner per ready source lane and return asynchronously."""

    from cuda.bindings import driver as cuda
    import deep_gemm

    deep_gemm_num_sms = os.getenv("SGLANG_DEEPEP_STREAMING_DEEPGEMM_NUM_SMS")
    original_deep_gemm_num_sms = deep_gemm.get_num_sms()
    if deep_gemm_num_sms is not None:
        deep_gemm.set_num_sms(int(deep_gemm_num_sms))

    lanes, lane_capacity, _ = dispatch.x.shape
    if streams is None:
        streams = tuple(torch.cuda.Stream(priority=0) for _ in range(lanes))
    if len(streams) != lanes:
        raise ValueError(f"expected {lanes} lane streams, got {len(streams)}")
    if drain_stream is None:
        drain_stream = torch.cuda.Stream(priority=0)
    # Resolve the protocol before queuing any device work.  A partially
    # submitted generation cannot safely fall back to the old all-lane ACK.
    release_streaming_lane = _require_per_lane_release(dispatch.buffer)

    # Keep the source stream ordered behind transport completion, but do not
    # synchronize the host here. Each lane stream consumes its own generation
    # doorbell below; alternating ElasticBuffers prevent the next generation
    # from reusing this slot until its ingress/return consumers have drained.
    # This preserves the attention-to-MoE readiness overlap on SM100.

    lane_output = torch.empty(
        dispatch.x.shape, dtype=torch.bfloat16, device=dispatch.x.device
    )
    returned: list[torch.cuda.Event] = []
    seq_stride_bytes = (
        dispatch.pack_done_seq.stride(0) * dispatch.pack_done_seq.element_size()
    )
    source_stream = torch.cuda.current_stream(dispatch.x.device)
    sync_baseline = is_deepep_v2_sync_baseline_enabled()
    all_lanes_ready = None
    if sync_baseline:
        # The baseline uses the exact same ElasticBuffer transport, lane-local
        # tensors, expert kernels, and source-local combine as the async path.
        # Only scheduling changes: every lane waits for the slowest source lane.
        for lane in range(lanes):
            _check_cuda_driver(
                cuda.cuStreamWaitValue64(
                    cuda.CUstream(source_stream.cuda_stream),
                    cuda.CUdeviceptr(
                        dispatch.pack_done_seq.data_ptr()
                        + lane * seq_stride_bytes
                    ),
                    dispatch.generation,
                    int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ),
                ),
                (
                    "wait for all DeepEP V2 lanes before synchronous baseline "
                    f"generation {dispatch.generation}"
                ),
                cuda,
            )
        all_lanes_ready = torch.cuda.Event()
        all_lanes_ready.record(source_stream)
    metadata_capacity = dispatch.src_metadata.size(0) // lanes
    dispatch_tensors = (
        dispatch.x,
        dispatch.sf,
        dispatch.route_weights,
        dispatch.src_metadata,
        dispatch.expert_psum,
        dispatch.pack_done_seq,
    )
    timeline_enabled = timeline_context is not None
    if timeline_enabled != (timeline_origin is not None):
        raise ValueError("timeline context and origin must be provided together")
    lane_timeline: list[dict[str, Any]] = []
    dispatch_done = None
    if timeline_enabled:
        profile_stream = torch.cuda.Stream(priority=0)
        dispatch_done = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(profile_stream):
            dispatch.transport_event.current_stream_wait()
            dispatch_done.record(profile_stream)

    host_gate_default = (
        "1" if torch.cuda.get_device_capability(dispatch.x.device)[0] >= 10 else "0"
    )
    host_lane_gate = (
        all_lanes_ready is None
        and os.getenv(
            "SGLANG_DEEPEP_STREAMING_HOST_LANE_GATE", host_gate_default
        ) == "1"
    )
    lane_order = range(lanes)
    if host_lane_gate:
        lane_ready_events = []
        for lane, stream in enumerate(streams):
            _check_cuda_driver(
                cuda.cuStreamWaitValue64(
                    cuda.CUstream(stream.cuda_stream),
                    cuda.CUdeviceptr(
                        dispatch.pack_done_seq.data_ptr()
                        + lane * seq_stride_bytes
                    ),
                    dispatch.generation,
                    int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ),
                ),
                f"wait for DeepEP lane {lane} generation {dispatch.generation}",
                cuda,
            )
            lane_ready = torch.cuda.Event()
            lane_ready.record(stream)
            lane_ready_events.append(lane_ready)

        def iter_ready_lanes():
            pending = set(range(lanes))
            while pending:
                made_progress = False
                for lane in tuple(pending):
                    if lane_ready_events[lane].query():
                        pending.remove(lane)
                        made_progress = True
                        yield lane
                if not made_progress:
                    time.sleep(0)

        lane_order = iter_ready_lanes()

    for lane in lane_order:
        stream = streams[lane]
        if all_lanes_ready is None and not host_lane_gate:
            _check_cuda_driver(
                cuda.cuStreamWaitValue64(
                    cuda.CUstream(stream.cuda_stream),
                    cuda.CUdeviceptr(
                        dispatch.pack_done_seq.data_ptr()
                        + lane * seq_stride_bytes
                    ),
                    dispatch.generation,
                    int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ),
                ),
                f"wait for DeepEP lane {lane} generation {dispatch.generation}",
                cuda,
            )
        elif all_lanes_ready is not None:
            stream.wait_event(all_lanes_ready)
        metadata = dispatch.src_metadata.narrow(
            0, lane * metadata_capacity, metadata_capacity
        )
        with torch.cuda.stream(stream):
            gemm_start = (
                torch.cuda.Event(enable_timing=True) if timeline_enabled else None
            )
            gemm_done = (
                torch.cuda.Event(enable_timing=True) if timeline_enabled else None
            )
            return_start = (
                torch.cuda.Event(enable_timing=True) if timeline_enabled else None
            )
            return_done = (
                torch.cuda.Event(enable_timing=True) if timeline_enabled else None
            )
            if gemm_start is not None:
                gemm_start.record(stream)
            transient_tensors = lane_compute(lane, lane_output[lane])
            if gemm_done is not None:
                gemm_done.record(stream)
            dispatch.transport_event.current_stream_wait()
            masked_route_weight_mul_(
                lane_output[lane],
                dispatch.route_weights[lane],
                dispatch.expert_psum[lane, -1:],
            )
            if return_start is not None:
                return_start.record(stream)
            dispatch.buffer.streaming_combine_return(
                lane_output[lane],
                metadata,
                lane,
                dispatch.generation,
            )
            # Keep every lane-owned input (including psums, route weights, and
            # metadata) alive through its return. Early protocol-v2 ACK is safe
            # only when all of those fields are snapshotted; delaying it here
            # avoids cross-generation overwrite while retaining rank-ready
            # attention-to-MoE scheduling within this generation.
            release_streaming_lane(lane, dispatch.generation)
            if return_done is not None:
                return_done.record(stream)

            lane_finalized = torch.cuda.Event()
            lane_finalized.record(stream)
            returned.append(lane_finalized)
            if timeline_enabled:
                lane_timeline.append(
                    {
                        "gemm_start": gemm_start,
                        "gemm_done": gemm_done,
                        "return_start": return_start,
                        "return_done": return_done,
                        "compute_scope": "lane",
                        "compute_group_id": lane,
                    }
                )

        for tensor in (
            *dispatch_tensors,
            *persistent_tensors,
            *transient_tensors,
            lane_output,
        ):
            if tensor is not None:
                tensor.record_stream(stream)

    # Keep the reduce on the serving stream, but express producer readiness as
    # CUDA event dependencies. On SM100, launching the reduce immediately and
    # spinning inside the kernel on return doorbells can race forward progress
    # under sustained multi-generation load. This is GPU-only synchronization:
    # the four lane pipelines remain asynchronous and the host never blocks.
    for done in returned:
        source_stream.wait_event(done)
    reduce_start = torch.cuda.Event(enable_timing=True) if timeline_enabled else None
    reduce_done = torch.cuda.Event(enable_timing=True) if timeline_enabled else None
    if reduce_start is not None:
        reduce_start.record(source_stream)
    combined_x, source_ready = dispatch.buffer.streaming_combine_reduce(
        dispatch.source_topk_idx, dispatch.generation
    )
    if reduce_done is not None:
        reduce_done.record(source_stream)
    # Per-lane release calls publish the remote ACKs.  This drain protects only
    # local view/tensor lifetime: strict per-lane DeepEP records its event for
    # destroy, but deliberately does not make the next dispatch wait on it.
    # Reduce must be host-submitted before finalizing the outstanding view.
    with torch.cuda.stream(drain_stream):
        # The return buffer is shared by successive streaming generations.
        # Waiting only for lane-return producers lets a later dispatch reuse it
        # while this source-reduce consumer is still reading on source_stream.
        # Join the reduce event into the generation lifetime fence as well.
        source_ready.current_stream_wait()
        for done in returned:
            drain_stream.wait_event(done)
        dispatch.buffer.release_streaming_lane_view()
        epoch_drained = torch.cuda.Event()
        epoch_drained.record(drain_stream)

    if timeline_enabled:
        _emit_streaming_timeline(
            dispatch,
            timeline_context,
            timeline_origin,
            dispatch_done,
            lane_timeline,
            reduce_start,
            reduce_done,
        )

    if deep_gemm_num_sms is not None:
        deep_gemm.set_num_sms(original_deep_gemm_num_sms)
    return DeepEPStreamingLayerResult(
        output=combined_x,
        source_ready=source_ready,
        source_stream=source_stream,
        epoch_drained=epoch_drained,
        lane_return_done=tuple(returned),
        dispatch=dispatch,
    )



def _launch_streaming_moe_waves(
    dispatch: DeepEPStreamingDispatch,
    wave_compute: Callable[[int, int, torch.Tensor], Sequence[torch.Tensor]],
    persistent_tensors: Sequence[torch.Tensor],
    *,
    wave_size: int,
    route_weights_applied: bool = False,
    streams: Sequence[torch.cuda.Stream],
    drain_stream: torch.cuda.Stream,
    timeline_context: dict[str, Any] | None,
    timeline_origin: torch.cuda.Event | None,
) -> DeepEPStreamingLayerResult:
    """Run contiguous source-lane wavefronts with repeated expert weights.

    SM100 grouped GEMMs are inefficient when each source lane launches a
    separate persistent kernel. A wave keeps the first-half/second-half arrival
    overlap while amortizing W13/W2 over several contiguous source lanes.
    """

    from cuda.bindings import driver as cuda
    import deep_gemm

    lanes, lane_capacity, _ = dispatch.x.shape
    if wave_size <= 1 or lanes % wave_size != 0:
        raise ValueError(f"wave_size={wave_size} must evenly divide {lanes} lanes")
    if len(streams) != lanes:
        raise ValueError(f"expected {lanes} lane streams, got {len(streams)}")

    original_deep_gemm_num_sms = deep_gemm.get_num_sms()
    requested_sms = os.getenv("SGLANG_DEEPEP_STREAMING_DEEPGEMM_NUM_SMS")
    # ElasticBuffer's SM100 dispatch uses 24 SMs on EP8. Keep those SMs free
    # while a wave GEMM is resident; this is an implementation invariant rather
    # than a workload tuning knob.
    wave_gemm_sms = (
        int(requested_sms)
        if requested_sms is not None
        else max(1, original_deep_gemm_num_sms - 24)
    )
    deep_gemm.set_num_sms(wave_gemm_sms)

    release_streaming_lane = _require_per_lane_release(dispatch.buffer)
    source_stream = torch.cuda.current_stream(dispatch.x.device)
    lane_output = torch.empty(
        dispatch.x.shape, dtype=torch.bfloat16, device=dispatch.x.device
    )
    seq_stride_bytes = (
        dispatch.pack_done_seq.stride(0) * dispatch.pack_done_seq.element_size()
    )
    metadata_capacity = dispatch.src_metadata.size(0) // lanes
    wave_ranges = tuple(
        (start, start + wave_size) for start in range(0, lanes, wave_size)
    )
    wave_streams = tuple(streams[start] for start, _ in wave_ranges)

    sync_baseline = is_deepep_v2_sync_baseline_enabled()
    all_lanes_ready = None
    if sync_baseline:
        for lane in range(lanes):
            _check_cuda_driver(
                cuda.cuStreamWaitValue64(
                    cuda.CUstream(source_stream.cuda_stream),
                    cuda.CUdeviceptr(
                        dispatch.pack_done_seq.data_ptr() + lane * seq_stride_bytes
                    ),
                    dispatch.generation,
                    int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ),
                ),
                f"wait for all DeepEP V2 lanes generation {dispatch.generation}",
                cuda,
            )
        all_lanes_ready = torch.cuda.Event()
        all_lanes_ready.record(source_stream)

    timeline_enabled = timeline_context is not None
    if timeline_enabled != (timeline_origin is not None):
        raise ValueError("timeline context and origin must be provided together")
    lane_timeline: list[dict[str, Any]] = [
        {} for _ in range(lanes)
    ]
    dispatch_done = None
    if timeline_enabled:
        profile_stream = torch.cuda.Stream(priority=0)
        dispatch_done = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(profile_stream):
            dispatch.transport_event.current_stream_wait()
            dispatch_done.record(profile_stream)

    # Each source lane keeps an independent transport stream.  GEMMs are
    # coalesced by wave, while return and ingress ACK remain source-local so
    # buffer reuse is not coupled to the slowest lane in a wave.
    wave_ready_events: list[torch.cuda.Event] = []
    for (start, stop), wave_stream in zip(wave_ranges, wave_streams):
        ingress_ready = []
        for lane in range(start, stop):
            lane_stream = streams[lane]
            if all_lanes_ready is None:
                _check_cuda_driver(
                    cuda.cuStreamWaitValue64(
                        cuda.CUstream(lane_stream.cuda_stream),
                        cuda.CUdeviceptr(
                            dispatch.pack_done_seq.data_ptr()
                            + lane * seq_stride_bytes
                        ),
                        dispatch.generation,
                        int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ),
                    ),
                    f"wait for DeepEP wave lane {lane} generation {dispatch.generation}",
                    cuda,
                )
            else:
                lane_stream.wait_event(all_lanes_ready)
            with torch.cuda.stream(lane_stream):
                ready = torch.cuda.Event()
                ready.record(lane_stream)
                ingress_ready.append(ready)

        # The first lane stream doubles as the wave GEMM stream. Queue only a
        # lightweight event join before host polling; the persistent GEMM is
        # submitted after all ingress streams have made forward progress.
        with torch.cuda.stream(wave_stream):
            for ready in ingress_ready:
                wave_stream.wait_event(ready)
            wave_ready = torch.cuda.Event()
            wave_ready.record(wave_stream)
            wave_ready_events.append(wave_ready)

    if all_lanes_ready is None:
        def iter_ready_waves():
            pending = set(range(len(wave_ranges)))
            while pending:
                made_progress = False
                for wave_index in tuple(pending):
                    if wave_ready_events[wave_index].query():
                        pending.remove(wave_index)
                        made_progress = True
                        yield wave_index
                if not made_progress:
                    time.sleep(0)

        wave_order = iter_ready_waves()
    else:
        wave_order = range(len(wave_ranges))

    returned: list[torch.cuda.Event | None] = [None] * lanes
    dispatch_tensors = (
        dispatch.x,
        dispatch.sf,
        dispatch.route_weights,
        dispatch.src_metadata,
        dispatch.expert_psum,
        dispatch.pack_done_seq,
    )
    for wave_index in wave_order:
        start, stop = wave_ranges[wave_index]
        stream = wave_streams[wave_index]
        with torch.cuda.stream(stream):
            gemm_start = (
                torch.cuda.Event(enable_timing=True) if timeline_enabled else None
            )
            gemm_done = (
                torch.cuda.Event(enable_timing=True) if timeline_enabled else None
            )
            if gemm_start is not None:
                gemm_start.record(stream)
            transient_tensors = wave_compute(start, stop, lane_output[start:stop])
            if gemm_done is not None:
                gemm_done.record(stream)
            wave_compute_done = torch.cuda.Event()
            wave_compute_done.record(stream)

        for tensor in (*persistent_tensors, *transient_tensors):
            if tensor is not None:
                tensor.record_stream(stream)

        for lane in range(start, stop):
            lane_stream = streams[lane]
            lane_stream.wait_event(wave_compute_done)
            with torch.cuda.stream(lane_stream):
                dispatch.transport_event.current_stream_wait()
                if not route_weights_applied:
                    masked_route_weight_mul_(
                        lane_output[lane],
                        dispatch.route_weights[lane],
                        dispatch.expert_psum[lane, -1:],
                    )

            for tensor in (*dispatch_tensors, *persistent_tensors, lane_output):
                if tensor is not None:
                    tensor.record_stream(lane_stream)

        for lane in range(start, stop):
            lane_stream = streams[lane]
            with torch.cuda.stream(lane_stream):
                return_start = (
                    torch.cuda.Event(enable_timing=True)
                    if timeline_enabled
                    else None
                )
                return_done = (
                    torch.cuda.Event(enable_timing=True)
                    if timeline_enabled
                    else None
                )
                if return_start is not None:
                    return_start.record(lane_stream)
                metadata = dispatch.src_metadata.narrow(
                    0, lane * metadata_capacity, metadata_capacity
                )
                dispatch.buffer.streaming_combine_return(
                    lane_output[lane], metadata, lane, dispatch.generation
                )
                release_streaming_lane(lane, dispatch.generation)
                if return_done is not None:
                    return_done.record(lane_stream)
                lane_finalized = torch.cuda.Event()
                lane_finalized.record(lane_stream)
                returned[lane] = lane_finalized
                if timeline_enabled:
                    lane_timeline[lane] = {
                        "gemm_start": gemm_start,
                        "gemm_done": gemm_done,
                        "return_start": return_start,
                        "return_done": return_done,
                        "compute_scope": "wave",
                        "compute_group_id": wave_index,
                    }
            metadata.record_stream(lane_stream)

    if any(done is None for done in returned):
        raise RuntimeError("not every DeepEP streaming lane scheduled a return")
    completed_returns = tuple(done for done in returned if done is not None)

    for done in completed_returns:
        source_stream.wait_event(done)
    reduce_start = (
        torch.cuda.Event(enable_timing=True) if timeline_enabled else None
    )
    reduce_done = torch.cuda.Event(enable_timing=True) if timeline_enabled else None
    if reduce_start is not None:
        reduce_start.record(source_stream)
    combined_x, source_ready = dispatch.buffer.streaming_combine_reduce(
        dispatch.source_topk_idx, dispatch.generation
    )
    if reduce_done is not None:
        reduce_done.record(source_stream)

    with torch.cuda.stream(drain_stream):
        source_ready.current_stream_wait()
        for done in completed_returns:
            drain_stream.wait_event(done)
        dispatch.buffer.release_streaming_lane_view()
        epoch_drained = torch.cuda.Event()
        epoch_drained.record(drain_stream)

    if timeline_enabled:
        _emit_streaming_timeline(
            dispatch,
            timeline_context,
            timeline_origin,
            dispatch_done,
            lane_timeline,
            reduce_start,
            reduce_done,
        )

    deep_gemm.set_num_sms(original_deep_gemm_num_sms)
    return DeepEPStreamingLayerResult(
        output=combined_x,
        source_ready=source_ready,
        source_stream=source_stream,
        epoch_drained=epoch_drained,
        lane_return_done=completed_returns,
        dispatch=dispatch,
    )

def launch_bf16_streaming_moe(
    dispatch: DeepEPStreamingDispatch,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    streams: Sequence[torch.cuda.Stream] | None = None,
    drain_stream: torch.cuda.Stream | None = None,
    expected_m_per_expert: int | None = None,
    swiglu_limit: float | None = None,
    timeline_context: dict[str, Any] | None = None,
    timeline_origin: torch.cuda.Event | None = None,
) -> DeepEPStreamingLayerResult:
    """Run W13, SwiGLU, W2, weighted return, and source-local combine.

    Each lane stream waits directly on its generation-tagged device doorbell.
    The source reduce is submitted immediately on the caller's stream and waits
    on return doorbells in the GPU kernel, so no host or rank barrier is added.
    """

    import deep_gemm
    from sglang.jit_kernel.activation import silu_and_mul
    from sglang.jit_kernel.dsv4.moe import silu_and_mul_clamp

    if dispatch.x.dtype != torch.bfloat16:
        raise ValueError(
            "streaming dispatch currently supports BF16 activations only; "
            f"lane payload has dtype={dispatch.x.dtype}, shape={tuple(dispatch.x.shape)}, "
            f"source tokens={dispatch.source_topk_idx.size(0)}"
        )
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError("streaming MoE currently supports BF16 expert weights only")
    if expected_m_per_expert is not None and (
        isinstance(expected_m_per_expert, bool)
        or not isinstance(expected_m_per_expert, int)
        or expected_m_per_expert <= 0
    ):
        raise ValueError("expected_m_per_expert must be a positive integer")
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError("expert weights must have shape [experts, N, K]")

    lanes, lane_capacity, hidden = dispatch.x.shape
    local_experts, gate_up_width, w13_physical_k = w13_weight.shape
    w2_experts, output_width, w2_physical_k = w2_weight.shape
    packing = 1
    intermediate = w2_physical_k * packing
    if (
        w13_physical_k * packing != hidden
        or w2_experts != local_experts
        or gate_up_width != 2 * intermediate
        or output_width != hidden
        or dispatch.expert_psum.size(1) != local_experts
    ):
        raise ValueError(
            "streaming activation, psum, and expert weight shapes disagree"
        )

    gate_up = torch.empty(
        (lanes, lane_capacity, gate_up_width),
        dtype=torch.bfloat16,
        device=dispatch.x.device,
    )
    down_input = torch.empty(
        (lanes, lane_capacity, intermediate),
        dtype=torch.bfloat16,
        device=dispatch.x.device,
    )

    def lane_compute(lane: int, lane_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            dispatch.x[lane],
            w13_weight,
            gate_up[lane],
            dispatch.expert_psum[lane],
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
        )
        if swiglu_limit is None:
            silu_and_mul(gate_up[lane], down_input[lane])
        else:
            silu_and_mul_clamp(gate_up[lane], down_input[lane], swiglu_limit)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            down_input[lane],
            w2_weight,
            lane_output,
            dispatch.expert_psum[lane],
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
        )
        return ()

    return _launch_streaming_moe_lanes(
        dispatch,
        lane_compute,
        (gate_up, down_input, w13_weight, w2_weight),
        streams=streams,
        drain_stream=drain_stream,
        timeline_context=timeline_context,
        timeline_origin=timeline_origin,
    )


def launch_fp8_streaming_moe(
    dispatch: DeepEPStreamingDispatch,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: Sequence[int],
    *,
    is_fp4_expert: bool = False,
    streams: Sequence[torch.cuda.Stream] | None = None,
    drain_stream: torch.cuda.Stream | None = None,
    activation_stream: torch.cuda.Stream | None = None,
    swiglu_limit: float | None = None,
    expected_m_per_expert: int | None = None,
    timeline_context: dict[str, Any] | None = None,
    timeline_origin: torch.cuda.Event | None = None,
) -> DeepEPStreamingLayerResult:
    """Run a block-FP8 expert MLP directly over each ready source lane.

    Dispatch quantizes each source activation once and DeepEP preserves both
    the FP8 payload and its TMA-aligned scales in lane-local layout. The two
    grouped GEMMs consume the psum layout directly; no shadow pack or host
    count readback is introduced.
    """

    import deep_gemm
    from sglang.jit_kernel.dsv4.moe import (
        silu_and_mul_clamp,
        silu_and_mul_masked_post_quant,
        silu_and_mul_psum_post_quant,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers import deep_gemm_wrapper

    if dispatch.x.dtype != torch.float8_e4m3fn or dispatch.sf is None:
        raise ValueError("FP8 streaming MoE requires FP8 payload and scales")
    if dispatch.sf.dtype not in (torch.float32, torch.int32):
        raise ValueError("FP8 activation scales must be float32 or packed int32")
    expected_weight_dtype = torch.int8 if is_fp4_expert else torch.float8_e4m3fn
    if w13_weight.dtype != expected_weight_dtype:
        raise ValueError(
            f"streaming MoE requires {expected_weight_dtype} W13 weights"
        )
    if w2_weight.dtype != expected_weight_dtype:
        raise ValueError(
            f"streaming MoE requires {expected_weight_dtype} W2 weights"
        )
    if w13_scale.dtype not in (torch.float32, torch.int32) or w2_scale.dtype not in (
        torch.float32,
        torch.int32,
    ):
        raise ValueError(
            "block-FP8 expert scales must be float32 or packed UE8M0 int32"
        )
    if tuple(block_shape) != (128, 128):
        raise ValueError(
            "FP8 streaming MoE currently requires a [128, 128] weight block"
        )
    if expected_m_per_expert is not None and (
        isinstance(expected_m_per_expert, bool)
        or not isinstance(expected_m_per_expert, int)
        or expected_m_per_expert <= 0
    ):
        raise ValueError("expected_m_per_expert must be a positive integer")

    lanes, lane_capacity, hidden = dispatch.x.shape
    local_experts, gate_up_width, w13_physical_k = w13_weight.shape
    w2_experts, output_width, w2_physical_k = w2_weight.shape
    packing = 2 if is_fp4_expert else 1
    intermediate = w2_physical_k * packing
    if (
        w13_physical_k * packing != hidden
        or w2_experts != local_experts
        or gate_up_width != 2 * intermediate
        or output_width != hidden
        or dispatch.expert_psum.size(1) != local_experts
    ):
        raise ValueError(
            "streaming activation, psum, and expert weight shapes disagree"
        )

    block_n, block_k = block_shape
    weight_gran_k = 32 if is_fp4_expert else block_k
    if w13_scale.dtype == torch.int32:
        expected_w13_scale = (
            local_experts,
            gate_up_width,
            ((hidden + weight_gran_k - 1) // weight_gran_k + 3) // 4,
        )
    elif is_fp4_expert:
        expected_w13_scale = (
            local_experts,
            gate_up_width,
            (hidden + weight_gran_k - 1) // weight_gran_k,
        )
    else:
        expected_w13_scale = (
            local_experts,
            (gate_up_width + block_n - 1) // block_n,
            (hidden + block_k - 1) // block_k,
        )
    if w2_scale.dtype == torch.int32:
        expected_w2_scale = (
            local_experts,
            hidden,
            ((intermediate + weight_gran_k - 1) // weight_gran_k + 3) // 4,
        )
    elif is_fp4_expert:
        expected_w2_scale = (
            local_experts,
            hidden,
            (intermediate + weight_gran_k - 1) // weight_gran_k,
        )
    else:
        expected_w2_scale = (
            local_experts,
            (hidden + block_n - 1) // block_n,
            (intermediate + block_k - 1) // block_k,
        )
    if tuple(w13_scale.shape) != expected_w13_scale:
        raise ValueError(
            f"expected W13 scale shape {expected_w13_scale}, got {tuple(w13_scale.shape)}"
        )
    if tuple(w2_scale.shape) != expected_w2_scale:
        raise ValueError(
            f"expected W2 scale shape {expected_w2_scale}, got {tuple(w2_scale.shape)}"
        )
    dispatch_scale_width = (hidden + block_k - 1) // block_k
    if dispatch.sf.dtype == torch.int32:
        dispatch_scale_width = (dispatch_scale_width + 3) // 4
    if dispatch.sf.size(2) != dispatch_scale_width:
        raise ValueError("FP8 dispatch scale width does not match hidden size")

    use_tma_aligned_scales = (
        deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES
        or deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    )
    if use_tma_aligned_scales and dispatch.sf.stride(1) != 1:
        raise ValueError("FP8 lane scales are not TMA-aligned column-major")
    if lane_capacity % 128 != 0:
        raise ValueError("FP8 lane capacity must preserve 128-row alignment")

    gate_up = torch.empty(
        (lanes, lane_capacity, gate_up_width),
        dtype=torch.bfloat16,
        device=dispatch.x.device,
    )
    down_input = None
    down_input_scale = None
    packed_lane_quant = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    rank_merge_enabled = is_deepep_streaming_rank_merge_enabled()
    if rank_merge_enabled and not (
        is_fp4_expert
        and packed_lane_quant
        and dispatch.sf.dtype == torch.int32
    ):
        raise RuntimeError(
            "rank-merged streaming currently requires FP8 activations with "
            "packed UE8M0 scales and MXFP4 expert weights"
        )
    use_masked_activation = packed_lane_quant
    lane_down_input_buffer = None
    lane_down_scale_storage = None
    if use_masked_activation:
        if not rank_merge_enabled and activation_stream is None:
            activation_stream = torch.cuda.Stream(priority=0)
        lane_scale_groups = intermediate // block_k
        if lane_scale_groups % 4 != 0:
            raise ValueError("packed UE8M0 activation scale width must divide by four")
        lane_down_input_buffer = torch.empty(
            (lanes, lane_capacity, intermediate),
            dtype=torch.float8_e4m3fn,
            device=dispatch.x.device,
        )
        lane_down_scale_storage = torch.empty(
            (lanes, lane_scale_groups // 4, lane_capacity),
            dtype=torch.int32,
            device=dispatch.x.device,
        )
    elif swiglu_limit is not None:
        down_input = torch.empty(
            (lanes, lane_capacity, intermediate),
            dtype=torch.bfloat16,
            device=dispatch.x.device,
        )

    # Allocate the large rank-merge scratch while DeepEP transport is already
    # running. Allocating it inside wave_compute would put allocator latency
    # after the all-lane readiness join and directly on the critical path.
    rank_merged_x_buffer = None
    rank_merged_input_scale_storage = None
    rank_merged_output_buffer = None
    if rank_merge_enabled:
        total_lane_rows = lanes * lane_capacity
        rank_merged_x_buffer = torch.empty(
            (total_lane_rows, hidden),
            dtype=dispatch.x.dtype,
            device=dispatch.x.device,
        )
        rank_merged_input_scale_storage = torch.empty(
            (dispatch.sf.size(2), total_lane_rows),
            dtype=dispatch.sf.dtype,
            device=dispatch.x.device,
        )
        rank_merged_output_buffer = torch.empty(
            (total_lane_rows, hidden),
            dtype=torch.bfloat16,
            device=dispatch.x.device,
        )

    gemm_kwargs = (
        {"recipe_a": (1, 128), "recipe_b": (1, 32)}
        if is_fp4_expert
        else {}
    )

    def lane_compute(lane: int, lane_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (dispatch.x[lane], dispatch.sf[lane]),
            (w13_weight, w13_scale),
            gate_up[lane],
            dispatch.expert_psum[lane],
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
            **gemm_kwargs,
        )
        if use_masked_activation:
            assert lane_down_input_buffer is not None
            assert lane_down_scale_storage is not None
            assert activation_stream is not None
            lane_stream = torch.cuda.current_stream(dispatch.x.device)
            w13_done = torch.cuda.Event()
            w13_done.record(lane_stream)
            with torch.cuda.stream(activation_stream):
                dispatch.transport_event.current_stream_wait()
                activation_stream.wait_event(w13_done)
                lane_active_rows = dispatch.expert_psum[
                    lane : lane + 1, -1
                ].clamp(min=0, max=lane_capacity)
                silu_and_mul_masked_post_quant(
                    gate_up[lane : lane + 1],
                    lane_down_input_buffer[lane : lane + 1],
                    lane_down_scale_storage[lane : lane + 1],
                    block_k,
                    lane_active_rows,
                    scale_ue8m0=True,
                    topk=1,
                    transposed=True,
                    use_pdl=False,
                    swiglu_limit=swiglu_limit,
                )
                activation_done = torch.cuda.Event()
                activation_done.record(activation_stream)
            lane_stream.wait_event(activation_done)
            lane_down_input = lane_down_input_buffer[lane]
            lane_down_input_scale = lane_down_scale_storage[lane].transpose(0, 1)
            transient_tensors = (lane_active_rows,)
        elif swiglu_limit is None:
            lane_down_input, lane_down_input_scale = (
                sglang_per_token_group_quant_fp8(
                    gate_up[lane],
                    block_k,
                    column_major_scales=use_tma_aligned_scales,
                    scale_tma_aligned=use_tma_aligned_scales,
                    scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                    fuse_silu_and_mul=True,
                )
            )
            transient_tensors = (lane_down_input, lane_down_input_scale)
        else:
            assert down_input is not None
            silu_and_mul_clamp(gate_up[lane], down_input[lane], swiglu_limit)
            lane_down_input, lane_down_input_scale = (
                sglang_per_token_group_quant_fp8(
                    down_input[lane],
                    block_k,
                    column_major_scales=use_tma_aligned_scales,
                    scale_tma_aligned=use_tma_aligned_scales,
                    scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                    fuse_silu_and_mul=False,
                )
            )
            transient_tensors = (
                lane_down_input,
                lane_down_input_scale,
            )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (lane_down_input, lane_down_input_scale),
            (w2_weight, w2_scale),
            lane_output,
            dispatch.expert_psum[lane],
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
            **gemm_kwargs,
        )
        return transient_tensors

    wave_size = _resolve_streaming_wave_size(
        lanes=lanes,
        device_major=torch.cuda.get_device_capability(dispatch.x.device)[0],
        grouped_gemm=deep_gemm.m_grouped_fp8_gemm_nt_contiguous,
    )
    if rank_merge_enabled and wave_size <= 1:
        raise RuntimeError(
            "rank-merged streaming requires "
            "SGLANG_DEEPEP_STREAMING_WAVE_SIZE greater than one"
        )
    if wave_size > 1:
        if lanes % wave_size != 0:
            raise ValueError(f"wave_size={wave_size} must divide {lanes} lanes")
        if streams is None:
            streams = tuple(torch.cuda.Stream(priority=0) for _ in range(lanes))
        if drain_stream is None:
            drain_stream = torch.cuda.Stream(priority=0)

        wave_gemm_kwargs = dict(gemm_kwargs)
        if not rank_merge_enabled:
            wave_gemm_kwargs["repeat_weight_groups"] = True

        def wave_compute(
            start: int, stop: int, wave_output: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            wave_lanes = stop - start
            wave_rows = wave_lanes * lane_capacity
            if rank_merge_enabled:
                assert dispatch.sf is not None
                assert dispatch.route_weights is not None
                assert lane_down_input_buffer is not None
                assert lane_down_scale_storage is not None
                layout = build_rank_merged_expert_layout(
                    dispatch.expert_psum[start:stop].contiguous(),
                    lane_capacity,
                    alignment=128,
                )
                assert rank_merged_x_buffer is not None
                assert rank_merged_input_scale_storage is not None
                assert rank_merged_output_buffer is not None
                merged_row_start = start * lane_capacity
                merged_x = rank_merged_x_buffer.narrow(
                    0, merged_row_start, wave_rows
                )
                merged_input_scale_storage = (
                    rank_merged_input_scale_storage.narrow(
                        1, merged_row_start, wave_rows
                    )
                )
                merged_input_scale = merged_input_scale_storage.t()
                pack_rank_merged_rows(
                    dispatch.x[start:stop], merged_x, layout
                )
                pack_rank_merged_rows(
                    dispatch.sf[start:stop], merged_input_scale, layout
                )

                merged_gate_up = gate_up[start:stop].reshape(
                    wave_rows, gate_up_width
                )
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (merged_x, merged_input_scale),
                    (w13_weight, w13_scale),
                    merged_gate_up,
                    layout.expert_psum,
                    use_psum_layout=True,
                    expected_m_for_psum_layout=expected_m_per_expert,
                    **wave_gemm_kwargs,
                )
                scale_groups = intermediate // block_k
                merged_down_input = lane_down_input_buffer[
                    start:stop
                ].reshape(wave_rows, intermediate)
                merged_down_scale_storage = lane_down_scale_storage[
                    start:stop
                ].reshape(scale_groups // 4, wave_rows)
                merged_down_scale = merged_down_scale_storage.t()
                dispatch.transport_event.current_stream_wait()
                silu_and_mul_psum_post_quant(
                    merged_gate_up,
                    merged_down_input,
                    merged_down_scale,
                    block_k,
                    layout.expert_psum,
                    expert_alignment=128,
                    scale_ue8m0=True,
                    transposed=True,
                    swiglu_limit=swiglu_limit,
                    use_pdl=False,
                )
                merged_result = rank_merged_output_buffer.narrow(
                    0, merged_row_start, wave_rows
                )
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (merged_down_input, merged_down_scale),
                    (w2_weight, w2_scale),
                    merged_result,
                    layout.expert_psum,
                    use_psum_layout=True,
                    expected_m_for_psum_layout=expected_m_per_expert,
                    **wave_gemm_kwargs,
                )
                scatter_rank_merged_rows(
                    merged_result,
                    wave_output,
                    layout,
                    route_weights=dispatch.route_weights[start:stop],
                )
                return (
                    merged_x,
                    merged_input_scale_storage,
                    merged_result,
                    layout.source_starts,
                    layout.counts,
                    layout.destination_starts,
                    layout.expert_psum,
                )

            wave_x = dispatch.x[start:stop].reshape(wave_rows, hidden)
            # DeepEP stores each lane's packed UE8M0 scales column-major. One
            # transpose-copy produces the column-major scale matrix for the
            # whole wave without copying the much larger FP8 activation.
            wave_sf_storage = (
                dispatch.sf[start:stop].permute(2, 0, 1).contiguous()
            )
            wave_sf = wave_sf_storage.reshape(
                wave_sf_storage.size(0), wave_rows
            ).t()
            lane_offsets = (
                torch.arange(
                    wave_lanes,
                    dtype=dispatch.expert_psum.dtype,
                    device=dispatch.expert_psum.device,
                ).unsqueeze(1)
                * lane_capacity
            )
            wave_psum = (
                dispatch.expert_psum[start:stop] + lane_offsets
            ).reshape(-1)
            wave_gate_up_lanes = gate_up[start:stop]
            wave_gate_up = wave_gate_up_lanes.reshape(wave_rows, gate_up_width)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (wave_x, wave_sf),
                (w13_weight, w13_scale),
                wave_gate_up,
                wave_psum,
                use_psum_layout=True,
                expected_m_for_psum_layout=expected_m_per_expert,
                **wave_gemm_kwargs,
            )
            # The lane allocation is sized for worst-case routing (16K rows on
            # DSV4), while a typical lane has only about 4K active psum rows.
            # Keep the shape static for DeepGEMM but let the fused device kernel
            # skip every lane's inactive tail without a host count readback.
            wave_active_rows = dispatch.expert_psum[start:stop, -1].clamp(
                min=0, max=lane_capacity
            )
            wave_down_input_lanes = torch.empty(
                (wave_lanes, lane_capacity, intermediate),
                dtype=torch.float8_e4m3fn,
                device=dispatch.x.device,
            )
            scale_groups = intermediate // block_k
            packed_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            wave_down_scale_storage = torch.empty(
                (
                    (wave_lanes, scale_groups // 4, lane_capacity)
                    if packed_ue8m0
                    else (wave_lanes, lane_capacity, scale_groups)
                ),
                dtype=torch.int32 if packed_ue8m0 else torch.float32,
                device=dispatch.x.device,
            )
            dispatch.transport_event.current_stream_wait()
            silu_and_mul_masked_post_quant(
                wave_gate_up_lanes,
                wave_down_input_lanes,
                wave_down_scale_storage,
                block_k,
                wave_active_rows,
                scale_ue8m0=packed_ue8m0,
                topk=wave_lanes,
                transposed=packed_ue8m0,
                use_pdl=False,
                swiglu_limit=swiglu_limit,
            )
            wave_down_lane_sf = (
                wave_down_scale_storage.transpose(-1, -2)
                if packed_ue8m0
                else wave_down_scale_storage
            )
            if packed_ue8m0:
                wave_down_sf_storage = (
                    wave_down_lane_sf.permute(2, 0, 1).contiguous()
                )
                wave_down_sf = wave_down_sf_storage.reshape(
                    wave_down_sf_storage.size(0), wave_rows
                ).t()
            else:
                wave_down_sf_storage = wave_down_scale_storage
                wave_down_sf = wave_down_scale_storage.reshape(
                    wave_rows, scale_groups
                )
            wave_down_input = wave_down_input_lanes.reshape(
                wave_rows, intermediate
            )
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (wave_down_input, wave_down_sf),
                (w2_weight, w2_scale),
                wave_output.reshape(wave_rows, hidden),
                wave_psum,
                use_psum_layout=True,
                expected_m_for_psum_layout=expected_m_per_expert,
                **wave_gemm_kwargs,
            )
            return (
                wave_sf_storage,
                lane_offsets,
                wave_psum,
                wave_active_rows,
                wave_down_input,
                wave_down_scale_storage,
                wave_down_sf_storage,
            )

        return _launch_streaming_moe_waves(
            dispatch,
            wave_compute,
            (
                gate_up,
                down_input,
                down_input_scale,
                lane_down_input_buffer,
                lane_down_scale_storage,
                rank_merged_x_buffer,
                rank_merged_input_scale_storage,
                rank_merged_output_buffer,
                w13_weight,
                w2_weight,
                w13_scale,
                w2_scale,
            ),
            wave_size=wave_size,
            route_weights_applied=rank_merge_enabled,
            streams=streams,
            drain_stream=drain_stream,
            timeline_context=timeline_context,
            timeline_origin=timeline_origin,
        )

    return _launch_streaming_moe_lanes(
        dispatch,
        lane_compute,
        (
            gate_up,
            down_input,
            down_input_scale,
            lane_down_input_buffer,
            lane_down_scale_storage,
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
        ),
        streams=streams,
        drain_stream=drain_stream,
        timeline_context=timeline_context,
        timeline_origin=timeline_origin,
    )
