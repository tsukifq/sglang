import pytest
import torch

from sglang.srt.layers.moe.deepep_streaming_kernels import (
    build_rank_merged_expert_layout,
    pack_rank_merged_rows,
    scatter_rank_merged_rows,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(
    est_time=5, stage="base-b-kernel-unit", runner_config="4-gpu-b200"
)


def _valid_row_copies(layout):
    source_starts = layout.source_starts.cpu()
    destination_starts = layout.destination_starts.cpu()
    counts = layout.counts.cpu()
    for lane in range(counts.size(0)):
        for expert in range(counts.size(1)):
            count = int(counts[lane, expert])
            yield (
                lane,
                int(source_starts[lane, expert]),
                int(destination_starts[lane, expert]),
                count,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rank_merged_layout_pack_and_weighted_scatter():
    psum = torch.tensor(
        ((2, 4, 7), (1, 6, 8), (0, 1, 6)),
        dtype=torch.int32,
        device="cuda",
    )
    layout = build_rank_merged_expert_layout(psum, 16, alignment=4)
    torch.cuda.synchronize()

    assert layout.source_starts.cpu().tolist() == [
        [0, 4, 4],
        [0, 4, 8],
        [0, 0, 4],
    ]
    assert layout.counts.cpu().tolist() == [
        [2, 0, 3],
        [1, 2, 0],
        [0, 1, 2],
    ]
    assert layout.destination_starts.cpu().tolist() == [
        [0, 4, 8],
        [2, 4, 11],
        [3, 6, 11],
    ]
    assert layout.expert_psum.cpu().tolist() == [3, 7, 13]

    lane_values = torch.arange(
        3 * 16 * 7, dtype=torch.float32, device="cuda"
    ).reshape(3, 16, 7)
    merged_values = torch.full(
        (3 * 16, 7), -1.0, dtype=torch.float32, device="cuda"
    )
    pack_rank_merged_rows(lane_values, merged_values, layout)

    expected_merged = torch.full_like(merged_values, -1.0)
    for lane, source_start, destination_start, count in _valid_row_copies(layout):
        expected_merged[destination_start : destination_start + count].copy_(
            lane_values[lane, source_start : source_start + count]
        )
    torch.testing.assert_close(merged_values, expected_merged)

    route_weights = (
        torch.arange(3 * 16, dtype=torch.float32, device="cuda").reshape(3, 16)
        / 100
        + 0.5
    )
    lane_output = torch.full_like(lane_values, -1.0)
    scatter_rank_merged_rows(
        merged_values, lane_output, layout, route_weights=route_weights
    )
    expected_output = torch.full_like(lane_output, -1.0)
    for lane, source_start, destination_start, count in _valid_row_copies(layout):
        expected_output[lane, source_start : source_start + count].copy_(
            merged_values[destination_start : destination_start + count]
            * route_weights[lane, source_start : source_start + count, None]
        )
    torch.testing.assert_close(lane_output, expected_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rank_merged_pack_supports_column_major_scales():
    psum = torch.tensor(
        ((2, 4, 7), (1, 6, 8), (0, 1, 6)),
        dtype=torch.int32,
        device="cuda",
    )
    layout = build_rank_merged_expert_layout(psum, 16, alignment=4)
    storage = torch.arange(
        3 * 5 * 16, dtype=torch.int32, device="cuda"
    ).reshape(3, 5, 16)
    lane_scales = storage.transpose(1, 2)
    merged_storage = torch.full(
        (5, 3 * 16), -1, dtype=torch.int32, device="cuda"
    )
    merged_scales = merged_storage.t()
    pack_rank_merged_rows(lane_scales, merged_scales, layout)

    expected = torch.full_like(merged_scales, -1)
    for lane, source_start, destination_start, count in _valid_row_copies(layout):
        expected[destination_start : destination_start + count].copy_(
            lane_scales[lane, source_start : source_start + count]
        )
    assert torch.equal(merged_scales, expected)
