"""Private fixed-shape input submission, owned by a leased prepared plan.

The caller validates dynamic activation signatures each generation. Only fixed
scratch, layouts and launch metadata are retained, never source input tensors.
The existing Triton kernels and launch mechanism are unchanged.
"""
import triton

from sglang.srt.layers.moe.deepep_streaming_kernels import (
    _build_rank_merged_expert_layout_kernel,
    _remap_rank_merged_rows_kernel,
    _validate_rank_merged_layout_request,
    _validate_rank_merged_remap,
    _resolve_rank_merged_row_splits,
)


class _PreparedRankMergedInput:
    def __init__(self, psum, x, sf, layout, merged_x, merged_sf, row_splits):
        lanes, experts = _validate_rank_merged_layout_request(
            psum, layout.lane_capacity, layout.alignment, require_contiguous=False
        )
        if layout.lane_to_merged is not None or layout.merged_route_weights is not None:
            raise ValueError("static input submission requires plain rank-merged layout")
        fields = (layout.source_starts, layout.counts, layout.destination_starts)
        if any(t.shape != psum.shape or t.dtype != psum.dtype or t.device != psum.device for t in fields):
            raise ValueError("static input layout fields do not match psum")
        if (layout.expert_psum.shape != (experts,) or layout.expert_psum.dtype != psum.dtype
                or layout.expert_psum.device != psum.device):
            raise ValueError("static input merged psum does not match")
        row_splits = _resolve_rank_merged_row_splits(row_splits)
        self.layout = layout
        self.merged_x = merged_x
        self.merged_sf = merged_sf
        self._layout_launch = _build_rank_merged_expert_layout_kernel[(1,)]
        self._layout_args = (*fields, layout.expert_psum)
        self._layout_kwargs = dict(
            num_lanes=lanes, num_experts=experts, alignment=layout.alignment,
            BLOCK_L=triton.next_power_of_2(lanes), psum_lane_stride=psum.stride(0),
            psum_expert_stride=psum.stride(1), num_warps=1,
        )
        self._packs = tuple(self._prepare_pack(src, dst, layout, row_splits)
                            for src, dst in ((x, merged_x), (sf, merged_sf)))

    @staticmethod
    def _prepare_pack(src, dst, layout, row_splits):
        lanes, experts, columns = _validate_rank_merged_remap(src, dst, layout)
        launch = _remap_rank_merged_rows_kernel[
            (lanes * experts * row_splits, triton.cdiv(columns, 256))
        ]
        args = (dst, layout.source_starts, layout.destination_starts, layout.counts, dst, dst)
        kwargs = dict(
            source_lane_stride=src.stride(0), source_row_stride=src.stride(1),
            source_col_stride=src.stride(2), destination_lane_stride=0,
            destination_row_stride=dst.stride(0), destination_col_stride=dst.stride(1),
            route_lane_stride=0, route_row_stride=0, num_experts=experts,
            num_cols=columns, REVERSE=False, APPLY_ROUTE_WEIGHT=False,
            WRITE_LANE_TO_MERGED=False, lane_capacity=layout.lane_capacity,
            BLOCK_M=8, BLOCK_N=256, ROW_SPLITS=row_splits, num_warps=4,
        )
        return launch, args, kwargs

    def run(self, psum, x, sf):
        """Submit on the current stream, already fenced by this epoch's ready."""
        self._layout_launch(psum, *self._layout_args, **self._layout_kwargs)
        for source, (launch, args, kwargs) in zip((x, sf), self._packs):
            launch(source, *args, **kwargs)
