"""Inference-only lane-streaming DeepEP consumer.

This module is intentionally separate from the regular dispatcher/runner
formats.  The experimental path consumes the lane-local sidecar exported by
Async MoE's DeepEP fork, runs a complete BF16 or block-FP8 expert MLP per
independently ready source lane, and uses source-local streaming combine
kernels.  The regular DeepEP path remains the default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from sglang.srt.environ import envs

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


def _launch_streaming_moe_lanes(
    dispatch: DeepEPStreamingDispatch,
    lane_compute: Callable[[int, torch.Tensor], Sequence[torch.Tensor]],
    persistent_tensors: Sequence[torch.Tensor],
    *,
    streams: Sequence[torch.cuda.Stream] | None,
    drain_stream: torch.cuda.Stream | None,
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
            transient_tensors = lane_compute(lane, lane_output[lane])
            lane_output[lane].mul_(dispatch.route_weights[lane].unsqueeze(1))
            dispatch.buffer.streaming_combine_return(
                lane_output[lane], metadata, lane, dispatch.generation
            )
            done = torch.cuda.Event()
            done.record(stream)
            returned.append(done)

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
    combined_x, source_ready = dispatch.buffer.streaming_combine_reduce(
        dispatch.source_topk_idx, dispatch.generation
    )
    with torch.cuda.stream(drain_stream):
        for done in returned:
            drain_stream.wait_event(done)
        dispatch.buffer.release_streaming_lane_view()
        epoch_drained = torch.cuda.Event()
        epoch_drained.record(drain_stream)

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
) -> DeepEPStreamingLayerResult:
    """Run W13, SwiGLU, W2, weighted return, and source-local combine.

    Each lane stream waits directly on its generation-tagged device doorbell.
    The source reduce is submitted immediately on the caller's stream and waits
    on return doorbells in the GPU kernel, so no host or rank barrier is added.
    """

    import deep_gemm
    from sglang.jit_kernel.activation import silu_and_mul

    if dispatch.x.dtype != torch.bfloat16:
        raise ValueError("streaming dispatch currently supports BF16 activations only")
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
    )
