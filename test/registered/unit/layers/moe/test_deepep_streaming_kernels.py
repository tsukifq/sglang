import pytest
import torch

from sglang.srt.layers.moe.deepep_streaming_kernels import (
    allocate_rank_merged_expert_layout,
    build_rank_merged_expert_layout,
    pack_rank_merged_rows,
    scatter_rank_merged_rows,
    weight_rank_merged_rows_,
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
    expected_merged = torch.full(
        (3 * 16, 7), -1.0, dtype=torch.float32, device="cuda"
    )
    for lane, source_start, destination_start, count in _valid_row_copies(layout):
        expected_merged[destination_start : destination_start + count].copy_(
            lane_values[lane, source_start : source_start + count]
        )

    route_weights = (
        torch.arange(3 * 16, dtype=torch.float32, device="cuda").reshape(3, 16)
        / 100
        + 0.5
    )
    expected_output = torch.full_like(lane_values, -1.0)
    for lane, source_start, destination_start, count in _valid_row_copies(layout):
        expected_output[lane, source_start : source_start + count].copy_(
            expected_merged[destination_start : destination_start + count]
            * route_weights[lane, source_start : source_start + count, None]
        )

    for row_splits in (1, 2, 4, 8):
        merged_values = torch.full_like(expected_merged, -1.0)
        pack_rank_merged_rows(
            lane_values, merged_values, layout, row_splits=row_splits
        )
        torch.testing.assert_close(merged_values, expected_merged)
        lane_output = torch.full_like(lane_values, -1.0)
        scatter_rank_merged_rows(
            merged_values,
            lane_output,
            layout,
            route_weights=route_weights,
            row_splits=row_splits,
        )
        torch.testing.assert_close(lane_output, expected_output)

        preweighted = merged_values.clone()
        weight_rank_merged_rows_(
            preweighted,
            route_weights,
            layout,
            row_splits=row_splits,
        )
        expected_preweighted = torch.full_like(preweighted, -1.0)
        for lane, source_start, destination_start, count in _valid_row_copies(layout):
            expected_preweighted[
                destination_start : destination_start + count
            ].copy_(
                merged_values[destination_start : destination_start + count]
                * route_weights[lane, source_start : source_start + count, None]
            )
        for _, _, destination_start, count in _valid_row_copies(layout):
            torch.testing.assert_close(
                preweighted[destination_start : destination_start + count],
                expected_preweighted[destination_start : destination_start + count],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rank_merged_layout_workspace_reuse():
    first = torch.tensor(
        ((2, 4, 7), (1, 6, 8), (0, 1, 6)),
        dtype=torch.int32,
        device="cuda",
    )
    second = torch.tensor(
        ((1, 5, 9), (0, 4, 8), (2, 4, 5)),
        dtype=torch.int32,
        device="cuda",
    )
    noncontiguous_template = first.t().contiguous().t()
    assert not noncontiguous_template.is_contiguous()
    workspace = allocate_rank_merged_expert_layout(
        noncontiguous_template, 16, alignment=4
    )
    assert workspace.source_starts.is_contiguous()
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.source_starts,
            workspace.counts,
            workspace.destination_starts,
            workspace.expert_psum,
        )
    )
    result = build_rank_merged_expert_layout(
        first, 16, alignment=4, workspace=workspace
    )
    assert result is workspace
    first_counts = result.counts.clone()
    result = build_rank_merged_expert_layout(
        second, 16, alignment=4, workspace=workspace
    )
    torch.cuda.synchronize()
    assert result is workspace
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            result.source_starts,
            result.counts,
            result.destination_starts,
            result.expert_psum,
        )
    )
    assert not torch.equal(first_counts, result.counts)

    wrong = allocate_rank_merged_expert_layout(
        first[:, :2].contiguous(), 16, alignment=4
    )
    with pytest.raises(ValueError, match="workspace"):
        build_rank_merged_expert_layout(
            first, 16, alignment=4, workspace=wrong
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rank_merged_layout_builds_optional_return_maps():
    psum = torch.tensor(
        ((2, 4, 7), (1, 6, 8), (0, 1, 6)),
        dtype=torch.int32,
        device="cuda",
    )
    route_weights = (
        torch.arange(3, device="cuda", dtype=torch.float32).unsqueeze(1)
        * 100
        + torch.arange(16, device="cuda", dtype=torch.float32)
    ).contiguous()
    workspace = allocate_rank_merged_expert_layout(
        psum,
        16,
        alignment=4,
        with_lane_to_merged=True,
        with_merged_route_weights=True,
    )
    layout = build_rank_merged_expert_layout(
        psum,
        16,
        alignment=4,
        workspace=workspace,
        with_lane_to_merged=True,
        with_merged_route_weights=True,
        route_weights=route_weights,
    )
    torch.cuda.synchronize()
    assert layout.lane_to_merged is not None
    assert layout.merged_route_weights is not None
    inverse = layout.lane_to_merged.cpu()
    merged_weights = layout.merged_route_weights.cpu()
    for lane, source_start, destination_start, count in _valid_row_copies(layout):
        assert inverse[
            lane, source_start : source_start + count
        ].tolist() == list(range(destination_start, destination_start + count))
        torch.testing.assert_close(
            merged_weights[destination_start : destination_start + count],
            route_weights[
                lane, source_start : source_start + count
            ].cpu(),
        )

    deferred_workspace = allocate_rank_merged_expert_layout(
        psum, 16, alignment=4, with_lane_to_merged=True
    )
    deferred = build_rank_merged_expert_layout(
        psum,
        16,
        alignment=4,
        workspace=deferred_workspace,
        with_lane_to_merged=True,
        defer_lane_to_merged=True,
    )
    lane_values = torch.arange(
        3 * 16 * 7, dtype=torch.float32, device="cuda"
    ).reshape(3, 16, 7)
    merged_values = torch.empty(
        (3 * 16, 7), dtype=torch.float32, device="cuda"
    )
    pack_rank_merged_rows(
        lane_values,
        merged_values,
        deferred,
        write_lane_to_merged=True,
    )
    torch.cuda.synchronize()
    deferred_inverse = deferred.lane_to_merged.cpu()
    for lane, source_start, destination_start, count in _valid_row_copies(deferred):
        assert deferred_inverse[
            lane, source_start : source_start + count
        ].tolist() == list(range(destination_start, destination_start + count))

    no_inverse = allocate_rank_merged_expert_layout(
        psum, 16, alignment=4
    )
    with pytest.raises(ValueError, match="workspace"):
        build_rank_merged_expert_layout(
            psum,
            16,
            alignment=4,
            workspace=no_inverse,
            with_lane_to_merged=True,
        )

    weights_workspace = allocate_rank_merged_expert_layout(
        psum, 16, alignment=4, with_merged_route_weights=True
    )
    with pytest.raises(ValueError, match="route weights"):
        build_rank_merged_expert_layout(
            psum,
            16,
            alignment=4,
            workspace=weights_workspace,
            with_merged_route_weights=True,
        )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("view_kind", ["sidecar", "transposed", "stepped", "broadcast"])
def test_rank_merged_layout_strided_psum(dtype, view_kind):
    # Empty experts, alignment gaps, uneven sources; compare against hand-coded
    # expected counts and starts in addition to the contiguous implementation.
    values = torch.tensor([[2, 4, 7], [1, 6, 8], [0, 1, 6]], dtype=dtype, device="cuda")
    if view_kind == "sidecar":
        storage = torch.full((3, 5), 123456, dtype=dtype, device="cuda")
        psum = storage[:, 2:]
        psum.copy_(values)
    elif view_kind == "transposed":
        psum = values.t().contiguous().t()
    elif view_kind == "stepped":
        storage = torch.full((6, 6), 123456, dtype=dtype, device="cuda")
        psum = storage[::2, ::2]
        psum.copy_(values)
    else:
        psum = values[:1].expand(3, 3)
    layout = build_rank_merged_expert_layout(psum, 16, alignment=4)
    reference = build_rank_merged_expert_layout(psum.contiguous(), 16, alignment=4)
    for field in ("source_starts", "counts", "destination_starts", "expert_psum"):
        torch.testing.assert_close(getattr(layout, field), getattr(reference, field), rtol=0, atol=0)
    if view_kind != "broadcast":
        assert layout.counts.cpu().tolist() == [[2, 0, 3], [1, 2, 0], [0, 1, 2]]
        assert layout.expert_psum.cpu().tolist() == [3, 7, 13]
    # Same allocated workspace must observe new generation values, not a cached psum.
    if view_kind == "sidecar":
        psum.zero_()
        updated = build_rank_merged_expert_layout(psum, 16, alignment=4, workspace=layout)
        assert updated is layout
        assert updated.expert_psum.cpu().tolist() == [0, 0, 0]
