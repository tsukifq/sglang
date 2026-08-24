import inspect
import os

import pytest

from sglang.srt.layers.moe.deepep_streaming import (
    DeepEPStreamingDispatch,
    _lane_layout_from_psum,
    _launch_streaming_moe_lanes,
    configure_deepep_streaming_environment,
    launch_fp8_streaming_moe,
)


@pytest.mark.parametrize(
    "name",
    [
        "EP_EXPERIMENTAL_STREAMING_COPY_SHADOW",
        "EP_EXPERIMENTAL_RANK_READY",
    ],
)
def test_streaming_environment_rejects_mixed_protocols(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match=name):
        configure_deepep_streaming_environment()


def test_streaming_environment_enables_required_protocol(monkeypatch):
    for name in (
        "EP_EXPERIMENTAL_STREAMING_COPY_SHADOW",
        "EP_EXPERIMENTAL_RANK_READY",
        "EP_EXPERIMENTAL_STREAMING_LANES",
        "EP_EXPERIMENTAL_STREAMING_LAYER",
        "EP_REUSE_NCCL_COMM",
        "NCCL_CUMEM_ENABLE",
    ):
        monkeypatch.delenv(name, raising=False)

    configure_deepep_streaming_environment()

    assert os.environ["EP_EXPERIMENTAL_STREAMING_LANES"] == "1"
    assert os.environ["EP_EXPERIMENTAL_STREAMING_LAYER"] == "1"
    assert os.environ["EP_REUSE_NCCL_COMM"] == "0"
    assert os.environ["NCCL_CUMEM_ENABLE"] == "1"


def test_streaming_environment_rejects_disabled_nccl_cumem(monkeypatch):
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "0")
    with pytest.raises(RuntimeError, match="NCCL_CUMEM_ENABLE=1"):
        configure_deepep_streaming_environment()


def test_streaming_dispatch_rejects_incomplete_runtime_view():
    with pytest.raises(ValueError, match="seven DeepEP lane-view fields"):
        DeepEPStreamingDispatch.from_runtime(
            buffer=object(),
            raw=(object(),),
            source_topk_idx=None,
            transport_handle=object(),
            transport_event=object(),
        )


def test_streaming_layer_keeps_dependencies_on_device():
    source = inspect.getsource(_launch_streaming_moe_lanes)
    assert "cuStreamWaitValue64" in source
    assert "streaming_combine_return" in source
    assert "streaming_combine_reduce" in source
    assert ".synchronize(" not in source
    assert ".barrier(" not in source


def test_timeline_decodes_aligned_lane_psum_without_counting_holes():
    layout = _lane_layout_from_psum([3, 128, 133])

    assert layout["expert_starts"] == [0, 128, 128]
    assert layout["expert_rows"] == [3, 0, 5]
    assert layout["useful_rows"] == 8
    assert layout["active_span_rows"] == 133
    assert layout["alignment_hole_rows"] == 125
    assert layout["nonempty_experts"] == 2


def test_fp8_streaming_consumes_psum_layout_without_shadow_pack():
    source = inspect.getsource(launch_fp8_streaming_moe)
    assert source.count("m_grouped_fp8_gemm_nt_contiguous") == 2
    assert source.count("use_psum_layout=True") == 2
    assert "fuse_silu_and_mul=True" in source
    assert "m_indices" not in source
    assert "shadow" in source
