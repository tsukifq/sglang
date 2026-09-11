"""Small device-side kernels used by DeepEP streaming MoE."""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True, slots=True)
class RankMergedExpertLayout:
    """Device-resident map between lane-local and owner-rank expert rows."""

    source_starts: torch.Tensor
    counts: torch.Tensor
    destination_starts: torch.Tensor
    expert_psum: torch.Tensor
    lane_to_merged: torch.Tensor | None
    merged_route_weights: torch.Tensor | None
    lane_capacity: int
    alignment: int


@triton.jit
def _build_rank_merged_expert_layout_kernel(
    expert_psum_ptr,
    source_starts_ptr,
    counts_ptr,
    destination_starts_ptr,
    merged_psum_ptr,
    num_lanes: tl.constexpr,
    num_experts: tl.constexpr,
    alignment: tl.constexpr,
    BLOCK_L: tl.constexpr,
    psum_lane_stride: tl.constexpr = -1,
    psum_expert_stride: tl.constexpr = 1,
):
    lane_stride = num_experts if psum_lane_stride < 0 else psum_lane_stride
    lanes = tl.arange(0, BLOCK_L)
    lane_mask = lanes < num_lanes
    merged_start = 0
    for expert in tl.static_range(0, num_experts):
        end = tl.load(
            expert_psum_ptr + lanes * lane_stride + expert * psum_expert_stride,
            mask=lane_mask,
            other=0,
        ).to(tl.int32)
        if expert == 0:
            source_start = tl.zeros((BLOCK_L,), dtype=tl.int32)
        else:
            previous_end = tl.load(
                expert_psum_ptr + lanes * lane_stride + (expert - 1) * psum_expert_stride,
                mask=lane_mask,
                other=0,
            ).to(tl.int32)
            source_start = (
                (previous_end + alignment - 1) // alignment * alignment
            )
        count = tl.maximum(end - source_start, 0)
        count = tl.where(lane_mask, count, 0)
        lane_prefix = tl.cumsum(count, axis=0) - count
        offsets = lanes * num_experts + expert
        tl.store(source_starts_ptr + offsets, source_start, mask=lane_mask)
        tl.store(counts_ptr + offsets, count, mask=lane_mask)
        tl.store(
            destination_starts_ptr + offsets,
            merged_start + lane_prefix,
            mask=lane_mask,
        )
        merged_end = merged_start + tl.sum(count, axis=0)
        tl.store(merged_psum_ptr + expert, merged_end)
        merged_start = (merged_end + alignment - 1) // alignment * alignment


@triton.jit
def _build_lane_to_merged_kernel(
    lane_to_merged_ptr,
    route_weights_ptr,
    merged_route_weights_ptr,
    source_starts_ptr,
    destination_starts_ptr,
    counts_ptr,
    lane_capacity: tl.constexpr,
    route_lane_stride: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    WRITE_LANE_TO_MERGED: tl.constexpr,
    WRITE_ROUTE_WEIGHTS: tl.constexpr,
):
    pair = tl.program_id(0)
    lane = pair // num_experts
    source_start = tl.load(source_starts_ptr + pair).to(tl.int32)
    destination_start = tl.load(destination_starts_ptr + pair).to(tl.int32)
    count = tl.load(counts_ptr + pair).to(tl.int32)
    for begin in range(0, count, BLOCK_M):
        rows = begin + tl.arange(0, BLOCK_M)
        mask = rows < count
        if WRITE_LANE_TO_MERGED:
            tl.store(
                lane_to_merged_ptr + lane * lane_capacity + source_start + rows,
                destination_start + rows,
                mask=mask,
            )
        if WRITE_ROUTE_WEIGHTS:
            weights = tl.load(
                route_weights_ptr
                + lane * route_lane_stride
                + source_start
                + rows,
                mask=mask,
            )
            tl.store(
                merged_route_weights_ptr + destination_start + rows,
                weights,
                mask=mask,
            )


def _build_lane_to_merged(
    layout: RankMergedExpertLayout,
    route_weights: torch.Tensor | None = None,
) -> None:
    if layout.lane_to_merged is None and layout.merged_route_weights is None:
        return
    lanes, experts = layout.counts.shape
    if layout.merged_route_weights is not None:
        if route_weights is None:
            raise ValueError("merged route weights require lane route weights")
        if (
            route_weights.shape != (lanes, layout.lane_capacity)
            or route_weights.dtype != torch.float32
            or not route_weights.is_contiguous()
            or route_weights.device != layout.counts.device
        ):
            raise ValueError(
                "route weights must be contiguous float32 [lanes, lane_capacity]"
            )
    lane_to_merged = (
        layout.lane_to_merged
        if layout.lane_to_merged is not None
        else layout.counts
    )
    merged_route_weights = (
        layout.merged_route_weights
        if layout.merged_route_weights is not None
        else layout.counts
    )
    route_weights_ptr = route_weights if route_weights is not None else layout.counts
    _build_lane_to_merged_kernel[(lanes * experts,)](
        lane_to_merged,
        route_weights_ptr,
        merged_route_weights,
        layout.source_starts,
        layout.destination_starts,
        layout.counts,
        lane_capacity=layout.lane_capacity,
        route_lane_stride=(
            route_weights.stride(0) if route_weights is not None else 0
        ),
        num_experts=experts,
        BLOCK_M=128,
        WRITE_LANE_TO_MERGED=layout.lane_to_merged is not None,
        WRITE_ROUTE_WEIGHTS=layout.merged_route_weights is not None,
        num_warps=4,
    )


def _validate_rank_merged_layout_request(
    expert_psum: torch.Tensor,
    lane_capacity: int,
    alignment: int,
    *,
    require_contiguous: bool = True,
) -> tuple[int, int]:
    if expert_psum.ndim != 2 or (
        require_contiguous and not expert_psum.is_contiguous()
    ):
        raise ValueError("expert_psum must be a contiguous [lanes, experts] tensor")
    if not expert_psum.is_cuda:
        raise ValueError("expert_psum must be CUDA-resident")
    if expert_psum.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert_psum must contain integer offsets")
    if lane_capacity <= 0:
        raise ValueError("lane_capacity must be positive")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")

    lanes, experts = expert_psum.shape
    if lanes <= 0 or experts <= 0:
        raise ValueError("expert_psum must contain at least one lane and expert")
    return lanes, experts


def allocate_rank_merged_expert_layout(
    expert_psum: torch.Tensor,
    lane_capacity: int,
    *,
    alignment: int = 128,
    with_lane_to_merged: bool = False,
    with_merged_route_weights: bool = False,
) -> RankMergedExpertLayout:
    """Allocate reusable device outputs for a rank-merged layout."""

    _, experts = _validate_rank_merged_layout_request(
        expert_psum, lane_capacity, alignment, require_contiguous=False
    )
    output_kwargs = {
        "dtype": expert_psum.dtype,
        "device": expert_psum.device,
    }
    source_starts = torch.empty(expert_psum.shape, **output_kwargs)
    counts = torch.empty(expert_psum.shape, **output_kwargs)
    destination_starts = torch.empty(expert_psum.shape, **output_kwargs)
    merged_psum = torch.empty(
        (experts,), dtype=expert_psum.dtype, device=expert_psum.device
    )
    lane_to_merged = (
        torch.empty(
            expert_psum.size(0),
            lane_capacity,
            dtype=expert_psum.dtype,
            device=expert_psum.device,
        )
        if with_lane_to_merged
        else None
    )
    merged_route_weights = (
        torch.empty(
            expert_psum.size(0) * lane_capacity,
            dtype=torch.float32,
            device=expert_psum.device,
        )
        if with_merged_route_weights
        else None
    )
    return RankMergedExpertLayout(
        source_starts=source_starts,
        counts=counts,
        destination_starts=destination_starts,
        expert_psum=merged_psum,
        lane_to_merged=lane_to_merged,
        merged_route_weights=merged_route_weights,
        lane_capacity=lane_capacity,
        alignment=alignment,
    )


def build_rank_merged_expert_layout(
    expert_psum: torch.Tensor,
    lane_capacity: int,
    *,
    alignment: int = 128,
    workspace: RankMergedExpertLayout | None = None,
    with_lane_to_merged: bool = False,
    with_merged_route_weights: bool = False,
    route_weights: torch.Tensor | None = None,
    defer_lane_to_merged: bool = False,
) -> RankMergedExpertLayout:
    """Merge equal experts across source lanes without a host count readback."""

    lanes, experts = _validate_rank_merged_layout_request(
        expert_psum, lane_capacity, alignment, require_contiguous=False
    )
    if workspace is None:
        workspace = allocate_rank_merged_expert_layout(
            expert_psum,
            lane_capacity,
            alignment=alignment,
            with_lane_to_merged=with_lane_to_merged,
            with_merged_route_weights=with_merged_route_weights,
        )
    else:
        fields = (
            workspace.source_starts,
            workspace.counts,
            workspace.destination_starts,
        )
        if (
            workspace.lane_capacity != lane_capacity
            or workspace.alignment != alignment
            or (with_lane_to_merged and workspace.lane_to_merged is None)
            or (
                with_merged_route_weights
                and workspace.merged_route_weights is None
            )
            or (workspace.lane_to_merged is not None and workspace.lane_to_merged.shape != (lanes, lane_capacity))
            or (
                workspace.merged_route_weights is not None
                and (
                    workspace.merged_route_weights.shape
                    != (lanes * lane_capacity,)
                    or workspace.merged_route_weights.dtype != torch.float32
                    or workspace.merged_route_weights.device
                    != expert_psum.device
                )
            )
            or any(t.shape != expert_psum.shape for t in fields)
            or workspace.expert_psum.shape != (experts,)
            or any(
                t.dtype != expert_psum.dtype
                for t in (*fields, workspace.expert_psum)
            )
            or any(
                t.device != expert_psum.device
                for t in (*fields, workspace.expert_psum)
            )
        ):
            raise ValueError("rank-merged layout workspace does not match the request")
    _build_rank_merged_expert_layout_kernel[(1,)](
        expert_psum,
        workspace.source_starts,
        workspace.counts,
        workspace.destination_starts,
        workspace.expert_psum,
        num_lanes=lanes,
        num_experts=experts,
        alignment=alignment,
        BLOCK_L=triton.next_power_of_2(lanes),
        psum_lane_stride=expert_psum.stride(0),
        psum_expert_stride=expert_psum.stride(1),
        num_warps=1,
    )
    if defer_lane_to_merged:
        if workspace.lane_to_merged is None:
            raise ValueError("deferred inverse-map build requires lane_to_merged")
        if workspace.merged_route_weights is not None:
            raise ValueError("merged route weights cannot use deferred inverse-map build")
    else:
        _build_lane_to_merged(workspace, route_weights)
    return workspace


@triton.jit
def _remap_rank_merged_rows_kernel(
    source_ptr,
    destination_ptr,
    source_starts_ptr,
    destination_starts_ptr,
    counts_ptr,
    lane_to_merged_ptr,
    route_weights_ptr,
    source_lane_stride: tl.constexpr,
    source_row_stride: tl.constexpr,
    source_col_stride: tl.constexpr,
    destination_lane_stride: tl.constexpr,
    destination_row_stride: tl.constexpr,
    destination_col_stride: tl.constexpr,
    route_lane_stride: tl.constexpr,
    route_row_stride: tl.constexpr,
    num_experts: tl.constexpr,
    num_cols: tl.constexpr,
    REVERSE: tl.constexpr,
    APPLY_ROUTE_WEIGHT: tl.constexpr,
    WRITE_LANE_TO_MERGED: tl.constexpr,
    lane_capacity: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROW_SPLITS: tl.constexpr,
):
    work = tl.program_id(0)
    pair = work // ROW_SPLITS
    row_split = work - pair * ROW_SPLITS
    lane = pair // num_experts
    expert = pair - lane * num_experts
    column_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = column_offsets < num_cols
    layout_offset = lane * num_experts + expert
    lane_start = tl.load(source_starts_ptr + layout_offset).to(tl.int32)
    merged_start = tl.load(destination_starts_ptr + layout_offset).to(tl.int32)
    count = tl.load(counts_ptr + layout_offset).to(tl.int32)
    if REVERSE:
        source_start = merged_start
        destination_start = lane_start
    else:
        source_start = lane_start
        destination_start = merged_start

    row_base = row_split * BLOCK_M
    while row_base < count:
        row_offsets = row_base + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < count
        source_offsets = (
            lane * source_lane_stride
            + (source_start + row_offsets[:, None]) * source_row_stride
            + column_offsets[None, :] * source_col_stride
        )
        destination_offsets = (
            lane * destination_lane_stride
            + (destination_start + row_offsets[:, None])
            * destination_row_stride
            + column_offsets[None, :] * destination_col_stride
        )
        mask = row_mask[:, None] & column_mask[None, :]
        values = tl.load(source_ptr + source_offsets, mask=mask)
        if APPLY_ROUTE_WEIGHT:
            weights = tl.load(
                route_weights_ptr
                + lane * route_lane_stride
                + (lane_start + row_offsets) * route_row_stride,
                mask=row_mask,
                other=0.0,
            )
            values = values * weights[:, None]
        tl.store(destination_ptr + destination_offsets, values, mask=mask)
        if WRITE_LANE_TO_MERGED:
            tl.store(
                lane_to_merged_ptr
                + lane * lane_capacity
                + lane_start
                + row_offsets,
                merged_start + row_offsets,
                mask=row_mask & (tl.program_id(1) == 0),
            )
        row_base += BLOCK_M * ROW_SPLITS


def _resolve_rank_merged_row_splits(row_splits: int | None) -> int:
    if row_splits is None:
        value = os.getenv("SGLANG_DEEPEP_STREAMING_REMAP_ROW_SPLITS", "1")
        try:
            row_splits = int(value)
        except ValueError as exc:
            raise ValueError(
                "SGLANG_DEEPEP_STREAMING_REMAP_ROW_SPLITS must be 1, 2, 4, or 8"
            ) from exc
    if isinstance(row_splits, bool) or row_splits not in (1, 2, 4, 8):
        raise ValueError("rank-merged row_splits must be 1, 2, 4, or 8")
    return row_splits


def _validate_rank_merged_remap(
    lane_tensor: torch.Tensor,
    merged_tensor: torch.Tensor,
    layout: RankMergedExpertLayout,
) -> tuple[int, int, int]:
    if lane_tensor.ndim != 3 or merged_tensor.ndim != 2:
        raise ValueError("rank-merged remap expects [lanes, rows, cols] and [rows, cols]")
    lanes, lane_capacity, columns = lane_tensor.shape
    if lanes != layout.counts.size(0) or lane_capacity != layout.lane_capacity:
        raise ValueError("lane tensor shape does not match rank-merged layout")
    if merged_tensor.shape != (lanes * lane_capacity, columns):
        raise ValueError("merged tensor must preserve total lane capacity")
    if lane_tensor.dtype != merged_tensor.dtype:
        raise ValueError("rank-merged source and destination dtypes must match")
    if (
        lane_tensor.device != merged_tensor.device
        or lane_tensor.device != layout.counts.device
    ):
        raise ValueError("rank-merged tensors must share one CUDA device")
    return lanes, layout.counts.size(1), columns


@triton.jit
def _rank_merged_scale_pack_rows_kernel(X, Y, S, D, C, LS: tl.constexpr, XC: tl.constexpr,
                YC: tl.constexpr, E: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    pair = tl.program_id(0)
    lane = pair // E
    count = tl.load(C + pair).to(tl.int32)
    src = tl.load(S + pair).to(tl.int32)
    dst = tl.load(D + pair).to(tl.int32)
    columns = tl.arange(0, BN)[:, None]
    rows = tl.arange(0, BM)[None, :]
    for begin in range(0, count, BM):
        r = begin + rows
        mask = (columns < N) & (r < count)
        values = tl.load(X + lane * LS + columns * XC + src + r, mask, other=0)
        tl.store(Y + columns * YC + dst + r, values, mask)


def _pack_rank_merged_scales_rows(lane_tensor, merged_tensor, layout):
    lanes, experts, columns = _validate_rank_merged_remap(lane_tensor, merged_tensor, layout)
    if (lane_tensor.dtype != torch.int32 or not 0 < columns <= 32
            or lane_tensor.stride(1) != 1 or merged_tensor.stride(0) != 1):
        raise ValueError('candidate requires int32 scales, 1..32 columns and contiguous rows')
    _rank_merged_scale_pack_rows_kernel[(lanes * experts,)](lane_tensor, merged_tensor, layout.source_starts,
        layout.destination_starts, layout.counts, lane_tensor.stride(0),
        lane_tensor.stride(2), merged_tensor.stride(1), experts, columns,
        128, triton.next_power_of_2(columns), num_warps=4)


def pack_rank_merged_rows(
    lane_tensor: torch.Tensor,
    merged_tensor: torch.Tensor,
    layout: RankMergedExpertLayout,
    *,
    row_splits: int | None = None,
    write_lane_to_merged: bool = False,
) -> None:
    """Pack valid lane-local expert rows into one owner-rank layout."""

    lanes, experts, columns = _validate_rank_merged_remap(
        lane_tensor, merged_tensor, layout
    )
    if write_lane_to_merged and layout.lane_to_merged is None:
        raise ValueError("inverse-map pack requires layout.lane_to_merged")
    # Only the column-major packed scale path is eligible. Payload and
    # unsupported scale layouts retain the existing remap implementation.
    if lane_tensor.dtype == torch.int32:
        scale_rows = os.getenv("SGLANG_DEEPEP_STREAMING_SCALE_PACK_ROWS", "0")
        if scale_rows not in ("0", "1"):
            raise ValueError("SGLANG_DEEPEP_STREAMING_SCALE_PACK_ROWS must be 0 or 1")
        if (not write_lane_to_merged and scale_rows == "1" and 0 < columns <= 32
                and lane_tensor.stride(1) == 1 and merged_tensor.stride(0) == 1):
            _pack_rank_merged_scales_rows(lane_tensor, merged_tensor, layout)
            return
    block_m, block_n = 8, 256
    row_splits = _resolve_rank_merged_row_splits(row_splits)
    _remap_rank_merged_rows_kernel[
        (lanes * experts * row_splits, triton.cdiv(columns, block_n))
    ](
        lane_tensor,
        merged_tensor,
        layout.source_starts,
        layout.destination_starts,
        layout.counts,
        layout.lane_to_merged if write_lane_to_merged else merged_tensor,
        merged_tensor,
        source_lane_stride=lane_tensor.stride(0),
        source_row_stride=lane_tensor.stride(1),
        source_col_stride=lane_tensor.stride(2),
        destination_lane_stride=0,
        destination_row_stride=merged_tensor.stride(0),
        destination_col_stride=merged_tensor.stride(1),
        route_lane_stride=0,
        route_row_stride=0,
        num_experts=experts,
        num_cols=columns,
        REVERSE=False,
        APPLY_ROUTE_WEIGHT=False,
        WRITE_LANE_TO_MERGED=write_lane_to_merged,
        lane_capacity=layout.lane_capacity,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ROW_SPLITS=row_splits,
        num_warps=4,
    )


def scatter_rank_merged_rows(
    merged_tensor: torch.Tensor,
    lane_tensor: torch.Tensor,
    layout: RankMergedExpertLayout,
    *,
    route_weights: torch.Tensor | None = None,
    row_splits: int | None = None,
) -> None:
    """Scatter owner-rank results and optionally apply lane-local routing weights."""

    lanes, experts, columns = _validate_rank_merged_remap(
        lane_tensor, merged_tensor, layout
    )
    if route_weights is not None:
        if route_weights.shape != lane_tensor.shape[:2]:
            raise ValueError("route_weights must have shape [lanes, lane_capacity]")
        if route_weights.device != lane_tensor.device:
            raise ValueError("route_weights must share the output CUDA device")
        route_pointer = route_weights
        route_lane_stride = route_weights.stride(0)
        route_row_stride = route_weights.stride(1)
    else:
        route_pointer = merged_tensor
        route_lane_stride = 0
        route_row_stride = 0
    block_m, block_n = 8, 256
    row_splits = _resolve_rank_merged_row_splits(row_splits)
    _remap_rank_merged_rows_kernel[
        (lanes * experts * row_splits, triton.cdiv(columns, block_n))
    ](
        merged_tensor,
        lane_tensor,
        layout.source_starts,
        layout.destination_starts,
        layout.counts,
        merged_tensor,
        route_pointer,
        source_lane_stride=0,
        source_row_stride=merged_tensor.stride(0),
        source_col_stride=merged_tensor.stride(1),
        destination_lane_stride=lane_tensor.stride(0),
        destination_row_stride=lane_tensor.stride(1),
        destination_col_stride=lane_tensor.stride(2),
        route_lane_stride=route_lane_stride,
        route_row_stride=route_row_stride,
        num_experts=experts,
        num_cols=columns,
        REVERSE=True,
        APPLY_ROUTE_WEIGHT=route_weights is not None,
        WRITE_LANE_TO_MERGED=False,
        lane_capacity=layout.lane_capacity,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ROW_SPLITS=row_splits,
        num_warps=4,
    )


@triton.jit
def _weight_rank_merged_rows_kernel(
    merged_ptr,
    source_starts_ptr,
    destination_starts_ptr,
    counts_ptr,
    route_weights_ptr,
    merged_row_stride: tl.constexpr,
    merged_col_stride: tl.constexpr,
    route_lane_stride: tl.constexpr,
    route_row_stride: tl.constexpr,
    num_experts: tl.constexpr,
    num_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROW_SPLITS: tl.constexpr,
):
    work = tl.program_id(0)
    pair = work // ROW_SPLITS
    row_split = work - pair * ROW_SPLITS
    lane = pair // num_experts
    column_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    column_mask = column_offsets < num_cols
    source_start = tl.load(source_starts_ptr + pair).to(tl.int32)
    destination_start = tl.load(destination_starts_ptr + pair).to(tl.int32)
    count = tl.load(counts_ptr + pair).to(tl.int32)
    row_base = row_split * BLOCK_M
    while row_base < count:
        row_offsets = row_base + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < count
        merged_offsets = (
            (destination_start + row_offsets[:, None]) * merged_row_stride
            + column_offsets[None, :] * merged_col_stride
        )
        mask = row_mask[:, None] & column_mask[None, :]
        values = tl.load(merged_ptr + merged_offsets, mask=mask)
        weights = tl.load(
            route_weights_ptr
            + lane * route_lane_stride
            + (source_start + row_offsets) * route_row_stride,
            mask=row_mask,
            other=0.0,
        )
        tl.store(merged_ptr + merged_offsets, values * weights[:, None], mask=mask)
        row_base += BLOCK_M * ROW_SPLITS


def weight_rank_merged_rows_(
    merged_tensor: torch.Tensor,
    route_weights: torch.Tensor,
    layout: RankMergedExpertLayout,
    *,
    row_splits: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
) -> None:
    """Apply lane-local route weights in place while retaining merged order."""

    if merged_tensor.ndim != 2 or not merged_tensor.is_contiguous():
        raise ValueError("merged tensor must be a contiguous 2D tensor")
    lanes, experts = layout.counts.shape
    if merged_tensor.size(0) != lanes * layout.lane_capacity:
        raise ValueError("merged tensor row count does not match the layout")
    if route_weights.shape != (lanes, layout.lane_capacity):
        raise ValueError("route weights must have shape [lanes, lane_capacity]")
    if (
        not merged_tensor.is_cuda
        or route_weights.device != merged_tensor.device
        or layout.counts.device != merged_tensor.device
    ):
        raise ValueError("merged rows, weights, and layout must share one CUDA device")
    if route_weights.dtype != torch.float32 or not route_weights.is_contiguous():
        raise ValueError("route weights must be contiguous float32")
    block_m = block_m or int(
        os.getenv("SGLANG_DEEPEP_STREAMING_PREWEIGHT_BLOCK_M", "4")
    )
    block_n = block_n or int(
        os.getenv("SGLANG_DEEPEP_STREAMING_PREWEIGHT_BLOCK_N", "512")
    )
    num_warps = num_warps or int(
        os.getenv("SGLANG_DEEPEP_STREAMING_PREWEIGHT_NUM_WARPS", "4")
    )
    if block_m not in (1, 2, 4, 8, 16, 32):
        raise ValueError("preweight block_m must be one of 1, 2, 4, 8, 16, 32")
    if block_n not in (64, 128, 256, 512):
        raise ValueError("preweight block_n must be one of 64, 128, 256, 512")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("preweight num_warps must be one of 1, 2, 4, 8")
    row_splits = _resolve_rank_merged_row_splits(row_splits)
    _weight_rank_merged_rows_kernel[
        (
            lanes * experts * row_splits,
            triton.cdiv(merged_tensor.size(1), block_n),
        )
    ](
        merged_tensor,
        layout.source_starts,
        layout.destination_starts,
        layout.counts,
        route_weights,
        merged_row_stride=merged_tensor.stride(0),
        merged_col_stride=merged_tensor.stride(1),
        route_lane_stride=route_weights.stride(0),
        route_row_stride=route_weights.stride(1),
        num_experts=experts,
        num_cols=merged_tensor.size(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ROW_SPLITS=row_splits,
        num_warps=num_warps,
    )


@triton.jit
def _masked_route_weight_mul_kernel(
    output_ptr,
    route_weight_ptr,
    active_rows_ptr,
    num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    active_rows = tl.load(active_rows_ptr).to(tl.int32)
    row_mask = (row_offsets < active_rows) & (row_offsets < num_rows)
    offsets = row_offsets[:, None] * num_cols + col_offsets[None, :]
    mask = row_mask[:, None] & (col_offsets[None, :] < num_cols)
    values = tl.load(output_ptr + offsets, mask=mask)
    weights = tl.load(route_weight_ptr + row_offsets, mask=row_mask)
    tl.store(output_ptr + offsets, values * weights[:, None], mask=mask)


def masked_route_weight_mul_(
    output: torch.Tensor,
    route_weights: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    """Multiply only the expert-packed rows published for one source lane."""

    if output.ndim != 2 or not output.is_contiguous():
        raise ValueError("output must be a contiguous 2D tensor")
    if route_weights.ndim != 1 or route_weights.numel() != output.size(0):
        raise ValueError("route weights must have one value per output row")
    if active_rows.numel() != 1 or active_rows.device != output.device:
        raise ValueError("active_rows must be one device scalar")
    if not output.is_cuda or route_weights.device != output.device:
        raise ValueError("output and route weights must share one CUDA device")
    if not route_weights.is_contiguous():
        raise ValueError("route weights must be contiguous")
    if active_rows.dtype not in (torch.int32, torch.int64):
        raise ValueError("active_rows must be an integer scalar")

    block_m, block_n = 4, 256
    grid = (
        triton.cdiv(output.size(0), block_m),
        triton.cdiv(output.size(1), block_n),
    )
    _masked_route_weight_mul_kernel[grid](
        output,
        route_weights,
        active_rows,
        num_rows=output.size(0),
        num_cols=output.size(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
