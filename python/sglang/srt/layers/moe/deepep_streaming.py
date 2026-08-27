"""Inference-only lane-streaming DeepEP consumer.

This module is intentionally separate from the regular dispatcher/runner
formats.  The experimental path consumes the lane-local sidecar exported by
Async MoE's DeepEP fork, runs a complete BF16 or block-FP8 expert MLP per
independently ready source lane, and uses source-local streaming combine
kernels.  The regular DeepEP path remains the default.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.profiling import (
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
)


def is_deepep_streaming_enabled() -> bool:
    return envs.SGLANG_ENABLE_DEEPEP_STREAMING.get()


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

    @classmethod
    def from_runtime(
        cls,
        *,
        buffer: Any,
        raw: tuple[Any, ...],
        source_topk_idx: torch.Tensor,
        transport_handle: Any,
        transport_event: Any,
    ) -> "DeepEPStreamingDispatch":
        if len(raw) != 7:
            raise ValueError(f"expected seven DeepEP lane-view fields, got {len(raw)}")
        view = cls(
            buffer,
            *raw,
            source_topk_idx,
            transport_handle,
            transport_event,
        )
        view.validate()
        return view

    def validate(self) -> None:
        if self.generation <= 0:
            raise ValueError("DeepEP streaming generation must be positive")
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
    lane_events: Sequence[dict[str, torch.cuda.Event]],
    reduce_start: torch.cuda.Event,
    reduce_done: torch.cuda.Event,
) -> None:
    """Queue one streaming sample for collection off the serving thread."""

    defer_started_ns = time.monotonic_ns()
    device = dispatch.x.device
    lane_events = tuple(dict(events) for events in lane_events)
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

    def collect() -> None:
        collector_started_ns = time.monotonic_ns()
        with torch.cuda.device(device):
            reduce_done.synchronize()
            dispatch_done.synchronize()
            for events in lane_events:
                events["return_done"].synchronize()
            output_wait_done_ns = time.monotonic_ns()

            profile_stream = torch.cuda.Stream(device=device, priority=0)
            clock_anchor = torch.cuda.Event(enable_timing=True)
            anchor_bracket_start_ns = time.monotonic_ns()
            clock_anchor.record(profile_stream)
            clock_anchor.synchronize()
            anchor_bracket_end_ns = time.monotonic_ns()

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
                "timed_cuda_event_count": 4 * len(lane_events) + 3,
            },
            "clock_alignment": {
                "method": (
                    "bracketed private-stream CUDA anchor projected to host "
                    "CLOCK_MONOTONIC"
                ),
                "anchor_host_monotonic_ns_midpoint": anchor_midpoint_ns,
                "anchor_bracket_start_ns": anchor_bracket_start_ns,
                "anchor_bracket_end_ns": anchor_bracket_end_ns,
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
                "moe_entry": aligned_timestamp(origin),
                "dispatch_transport_done": aligned_timestamp(dispatch_done),
                "combine_reduce_start": aligned_timestamp(reduce_start),
                "combine_reduce_done": aligned_timestamp(reduce_done),
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

    lanes, lane_capacity, _ = dispatch.x.shape
    if streams is None:
        streams = tuple(torch.cuda.Stream(priority=0) for _ in range(lanes))
    if len(streams) != lanes:
        raise ValueError(f"expected {lanes} lane streams, got {len(streams)}")
    if drain_stream is None:
        drain_stream = torch.cuda.Stream(priority=0)

    lane_output = torch.empty(
        dispatch.x.shape, dtype=torch.bfloat16, device=dispatch.x.device
    )
    returned: list[torch.cuda.Event] = []
    seq_stride_bytes = (
        dispatch.pack_done_seq.stride(0) * dispatch.pack_done_seq.element_size()
    )
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
    lane_timeline: list[dict[str, torch.cuda.Event]] = []
    dispatch_done = None
    if timeline_enabled:
        profile_stream = torch.cuda.Stream(priority=0)
        dispatch_done = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(profile_stream):
            dispatch.transport_event.current_stream_wait()
            dispatch_done.record(profile_stream)

    for lane, stream in enumerate(streams):
        _check_cuda_driver(
            cuda.cuStreamWaitValue64(
                cuda.CUstream(stream.cuda_stream),
                cuda.CUdeviceptr(
                    dispatch.pack_done_seq.data_ptr() + lane * seq_stride_bytes
                ),
                dispatch.generation,
                int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ),
            ),
            f"wait for DeepEP lane {lane} generation {dispatch.generation}",
            cuda,
        )
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
            lane_output[lane].mul_(dispatch.route_weights[lane].unsqueeze(1))
            if return_start is not None:
                return_start.record(stream)
            dispatch.buffer.streaming_combine_return(
                lane_output[lane], metadata, lane, dispatch.generation
            )
            done = return_done or torch.cuda.Event()
            done.record(stream)
            returned.append(done)
            if timeline_enabled:
                lane_timeline.append(
                    {
                        "gemm_start": gemm_start,
                        "gemm_done": gemm_done,
                        "return_start": return_start,
                        "return_done": return_done,
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

    # The reduce kernel waits only for this source rank's destination returns.
    # Keeping it on the serving stream makes the next operation's dependency
    # ordinary CUDA stream order rather than an inter-rank synchronization.
    source_stream = torch.cuda.current_stream(dispatch.x.device)
    reduce_start = torch.cuda.Event(enable_timing=True) if timeline_enabled else None
    reduce_done = torch.cuda.Event(enable_timing=True) if timeline_enabled else None
    if reduce_start is not None:
        reduce_start.record(source_stream)
    combined_x, source_ready = dispatch.buffer.streaming_combine_reduce(
        dispatch.source_topk_idx, dispatch.generation
    )
    if reduce_done is not None:
        reduce_done.record(source_stream)
    with torch.cuda.stream(drain_stream):
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

    return DeepEPStreamingLayerResult(
        output=combined_x,
        source_ready=source_ready,
        source_stream=source_stream,
        epoch_drained=epoch_drained,
        lane_return_done=tuple(returned),
        dispatch=dispatch,
    )


def launch_bf16_streaming_moe(
    dispatch: DeepEPStreamingDispatch,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    streams: Sequence[torch.cuda.Stream] | None = None,
    drain_stream: torch.cuda.Stream | None = None,
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

    if dispatch.x.dtype != torch.bfloat16:
        raise ValueError(
            "streaming dispatch currently supports BF16 activations only; "
            f"lane payload has dtype={dispatch.x.dtype}, shape={tuple(dispatch.x.shape)}, "
            f"source tokens={dispatch.source_topk_idx.size(0)}"
        )
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError("streaming MoE currently supports BF16 expert weights only")
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError("expert weights must have shape [experts, N, K]")

    lanes, lane_capacity, hidden = dispatch.x.shape
    local_experts, gate_up_width, w13_hidden = w13_weight.shape
    w2_experts, output_width, intermediate = w2_weight.shape
    if (
        w13_hidden != hidden
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
        )
        silu_and_mul(gate_up[lane], down_input[lane])
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            down_input[lane],
            w2_weight,
            lane_output,
            dispatch.expert_psum[lane],
            use_psum_layout=True,
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
    streams: Sequence[torch.cuda.Stream] | None = None,
    drain_stream: torch.cuda.Stream | None = None,
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
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers import deep_gemm_wrapper

    if dispatch.x.dtype != torch.float8_e4m3fn or dispatch.sf is None:
        raise ValueError("FP8 streaming MoE requires FP8 payload and scales")
    if dispatch.sf.dtype not in (torch.float32, torch.int32):
        raise ValueError("FP8 activation scales must be float32 or packed int32")
    if w13_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 streaming MoE requires e4m3 W13 weights")
    if w2_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 streaming MoE requires e4m3 W2 weights")
    if w13_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        raise ValueError("block-FP8 expert scales must be float32")
    if tuple(block_shape) != (128, 128):
        raise ValueError(
            "FP8 streaming MoE currently requires a [128, 128] weight block"
        )

    lanes, lane_capacity, hidden = dispatch.x.shape
    local_experts, gate_up_width, w13_hidden = w13_weight.shape
    w2_experts, output_width, intermediate = w2_weight.shape
    if (
        w13_hidden != hidden
        or w2_experts != local_experts
        or gate_up_width != 2 * intermediate
        or output_width != hidden
        or dispatch.expert_psum.size(1) != local_experts
    ):
        raise ValueError(
            "streaming activation, psum, and FP8 expert weight shapes disagree"
        )

    block_n, block_k = block_shape
    expected_w13_scale = (
        local_experts,
        (gate_up_width + block_n - 1) // block_n,
        (hidden + block_k - 1) // block_k,
    )
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
    if dispatch.sf.size(2) != (hidden + block_k - 1) // block_k:
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

    def lane_compute(lane: int, lane_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (dispatch.x[lane], dispatch.sf[lane]),
            (w13_weight, w13_scale),
            gate_up[lane],
            dispatch.expert_psum[lane],
            use_psum_layout=True,
        )
        down_input, down_input_scale = sglang_per_token_group_quant_fp8(
            gate_up[lane],
            block_k,
            column_major_scales=use_tma_aligned_scales,
            scale_tma_aligned=use_tma_aligned_scales,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            fuse_silu_and_mul=True,
        )
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (down_input, down_input_scale),
            (w2_weight, w2_scale),
            lane_output,
            dispatch.expert_psum[lane],
            use_psum_layout=True,
        )
        return down_input, down_input_scale

    return _launch_streaming_moe_lanes(
        dispatch,
        lane_compute,
        (gate_up, w13_weight, w2_weight, w13_scale, w2_scale),
        streams=streams,
        drain_stream=drain_stream,
        timeline_context=timeline_context,
        timeline_origin=timeline_origin,
    )
