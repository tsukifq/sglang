import inspect
import os

import pytest

from sglang.srt.layers.moe.deepep_streaming import (
    DeepEPStreamingDispatch,
    configure_deepep_streaming_environment,
    launch_bf16_streaming_moe,
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
    source = inspect.getsource(launch_bf16_streaming_moe)
    assert "cuStreamWaitValue64" in source
    assert "streaming_combine_return" in source
    assert "streaming_combine_reduce" in source
    assert ".synchronize(" not in source
    assert ".barrier(" not in source
