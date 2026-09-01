"""Small device-side kernels used by DeepEP streaming MoE."""

import torch
import triton
import triton.language as tl


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

