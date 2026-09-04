"""Small device-side kernels used by DeepEP streaming MoE."""

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
):
    lanes = tl.arange(0, BLOCK_L)
    lane_mask = lanes < num_lanes
    merged_start = 0
    for expert in tl.static_range(0, num_experts):
        end = tl.load(
            expert_psum_ptr + lanes * num_experts + expert,
            mask=lane_mask,
            other=0,
        ).to(tl.int32)
        if expert == 0:
            source_start = tl.zeros((BLOCK_L,), dtype=tl.int32)
        else:
            previous_end = tl.load(
                expert_psum_ptr + lanes * num_experts + expert - 1,
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


def build_rank_merged_expert_layout(
    expert_psum: torch.Tensor,
    lane_capacity: int,
    *,
    alignment: int = 128,
) -> RankMergedExpertLayout:
    """Merge equal experts across source lanes without a host count readback."""

    if expert_psum.ndim != 2 or not expert_psum.is_contiguous():
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
    source_starts = torch.empty_like(expert_psum)
    counts = torch.empty_like(expert_psum)
    destination_starts = torch.empty_like(expert_psum)
    merged_psum = torch.empty(
        (experts,), dtype=expert_psum.dtype, device=expert_psum.device
    )
    _build_rank_merged_expert_layout_kernel[(1,)](
        expert_psum,
        source_starts,
        counts,
        destination_starts,
        merged_psum,
        num_lanes=lanes,
        num_experts=experts,
        alignment=alignment,
        BLOCK_L=triton.next_power_of_2(lanes),
        num_warps=1,
    )
    return RankMergedExpertLayout(
        source_starts=source_starts,
        counts=counts,
        destination_starts=destination_starts,
        expert_psum=merged_psum,
        lane_capacity=lane_capacity,
        alignment=alignment,
    )


@triton.jit
def _remap_rank_merged_rows_kernel(
    source_ptr,
    destination_ptr,
    source_starts_ptr,
    destination_starts_ptr,
    counts_ptr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pair = tl.program_id(0)
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

    row_base = 0
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
        row_base += BLOCK_M


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


def pack_rank_merged_rows(
    lane_tensor: torch.Tensor,
    merged_tensor: torch.Tensor,
    layout: RankMergedExpertLayout,
) -> None:
    """Pack valid lane-local expert rows into one owner-rank layout."""

    lanes, experts, columns = _validate_rank_merged_remap(
        lane_tensor, merged_tensor, layout
    )
    block_m, block_n = 8, 256
    _remap_rank_merged_rows_kernel[(lanes * experts, triton.cdiv(columns, block_n))](
        lane_tensor,
        merged_tensor,
        layout.source_starts,
        layout.destination_starts,
        layout.counts,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )


def scatter_rank_merged_rows(
    merged_tensor: torch.Tensor,
    lane_tensor: torch.Tensor,
    layout: RankMergedExpertLayout,
    *,
    route_weights: torch.Tensor | None = None,
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
    _remap_rank_merged_rows_kernel[(lanes * experts, triton.cdiv(columns, block_n))](
        merged_tensor,
        lane_tensor,
        layout.source_starts,
        layout.destination_starts,
        layout.counts,
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
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
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

