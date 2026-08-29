import inspect
import os
from types import SimpleNamespace

import pytest

from sglang.srt.batch_overlap.operations import _resolve_tbo_child_contexts
from sglang.srt.layers.moe.deepep_streaming import (
    DeepEPStreamingDispatch,
    _lane_layout_from_psum,
    _launch_streaming_moe_lanes,
    _require_per_lane_release,
    configure_deepep_streaming_environment,
    launch_bf16_streaming_moe,
    launch_fp8_streaming_moe,
)
from sglang.srt.layers.moe.token_dispatcher import deepep as deepep_dispatcher_module
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPDispatcher,
    DeepEPStreamingBuffer,
    _DeepEPDispatcherImplNormal,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_moe_wavefront_slot,
)


@pytest.mark.parametrize(
    "name",
    [
        "EP_EXPERIMENTAL_STREAMING_COPY_SHADOW",
        "EP_EXPERIMENTAL_RANK_READY",
        "EP_EXPERIMENTAL_STREAMING_INSTRUMENTED_BULK",
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


def test_tbo_children_receive_distinct_moe_wavefront_slots():
    with forward_context(ForwardContext(attn_backend=object())):
        child_a, child_b = _resolve_tbo_child_contexts()
    assert child_a.moe_wavefront_slot == 0
    assert child_b.moe_wavefront_slot == 1
    with forward_context(child_b):
        assert get_moe_wavefront_slot() == 1


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
    assert "release_streaming_lane(lane, dispatch.generation)" in source
    assert ".synchronize(" not in source
    assert ".barrier(" not in source


def test_streaming_lane_ack_is_ordered_after_pack_wait_and_before_gemm():
    source = inspect.getsource(_launch_streaming_moe_lanes)

    pack_wait = source.index("cuStreamWaitValue64")
    release_lane = source.index("release_streaming_lane(lane, dispatch.generation)")
    lane_compute = source.index("lane_compute(lane, lane_output[lane])")
    combine_return = source.index("dispatch.buffer.streaming_combine_return(")

    assert pack_wait < release_lane < lane_compute < combine_return


def test_streaming_generation_owned_control_removes_deferred_psum_copy():
    source = inspect.getsource(_launch_streaming_moe_lanes)

    assert "expert_psum_snapshot" not in source


def test_streaming_view_finalize_is_after_reduce_and_lane_drain_waits():
    source = inspect.getsource(_launch_streaming_moe_lanes)

    reduce = source.index("dispatch.buffer.streaming_combine_reduce(")
    drain_join = source.index("with torch.cuda.stream(drain_stream):", reduce)
    lane_wait = source.index("drain_stream.wait_event(done)", drain_join)
    finalize = source.index("dispatch.buffer.release_streaming_lane_view()", lane_wait)
    epoch_drained = source.index("epoch_drained.record(drain_stream)", finalize)

    assert reduce < drain_join < lane_wait < finalize < epoch_drained


def test_streaming_lane_release_api_is_fail_closed():
    class LegacyBuffer:
        def release_streaming_lane_view(self):
            raise AssertionError("unsafe legacy fallback must not be called")

    with pytest.raises(RuntimeError, match="generation-aware.*release_streaming_lane"):
        _require_per_lane_release(LegacyBuffer())


def test_streaming_lane_release_api_returns_exact_callable():
    calls = []

    class Buffer:
        def get_streaming_lane_protocol_version(self):
            return 2

        def release_streaming_lane(self, source_rank, generation):
            calls.append((source_rank, generation))

        def release_streaming_lane_view(self):
            pass

    release_lane = _require_per_lane_release(Buffer())
    release_lane(3, 11)

    assert calls == [(3, 11)]


def test_streaming_lane_release_rejects_stale_native_runtime():
    class StaleRuntime:
        pass

    class SourceWrapper:
        runtime = StaleRuntime()

        def get_streaming_lane_protocol_version(self):
            return 2

        def release_streaming_lane(self, source_rank, generation):
            raise AssertionError("stale native runtime must be rejected first")

        def release_streaming_lane_view(self):
            pass

    with pytest.raises(RuntimeError, match="generation-aware.*release_streaming_lane"):
        _require_per_lane_release(SourceWrapper())


def test_streaming_lane_release_rejects_v1_before_transport():
    class V1Buffer:
        def get_streaming_lane_protocol_version(self):
            return 1

        def release_streaming_lane(self, source_rank, generation):
            raise AssertionError("v1 release must not be returned")

        def release_streaming_lane_view(self):
            pass

    with pytest.raises(RuntimeError, match="protocol v2 before transport"):
        _require_per_lane_release(V1Buffer())


def test_streaming_dispatch_preflights_release_api_before_transport():
    source = inspect.getsource(_DeepEPDispatcherImplNormal.dispatch_streaming)

    preflight = source.index("_require_per_lane_release(buffer)")
    transport = source.index("buffer.dispatch(", preflight)

    assert preflight < transport


def test_streaming_buffer_pool_builds_independent_wavefront_slots(monkeypatch):
    state = SimpleNamespace(buffers=None, signature=None)
    created = []

    class FakeElasticBuffer:
        def __init__(self, group, **kwargs):
            self.group = group
            self.kwargs = kwargs
            created.append(self)

    class FakeGroup:
        def size(self):
            return 8

    monkeypatch.setattr(
        DeepEPStreamingBuffer,
        "_state",
        classmethod(lambda cls: state),
    )
    monkeypatch.setattr(
        deepep_dispatcher_module, "ElasticBuffer", FakeElasticBuffer
    )

    slot0 = DeepEPStreamingBuffer.get_buffer(
        FakeGroup(), 4096, 8, 256, False, wavefront_slot=0, num_wavefront_slots=2
    )
    slot1 = DeepEPStreamingBuffer.get_buffer(
        FakeGroup(), 4096, 8, 256, False, wavefront_slot=1, num_wavefront_slots=2
    )

    assert len(created) == 2
    assert slot0 is created[0]
    assert slot1 is created[1]
    assert slot0 is not slot1
    assert state.buffers == (slot0, slot1)


def test_streaming_buffer_pool_rejects_slot_signature_drift(monkeypatch):
    state = SimpleNamespace(
        buffers=(object(), object()),
        signature=(8, 4096, 8, 256, False, 2),
    )

    class FakeGroup:
        def size(self):
            return 8

    monkeypatch.setattr(
        DeepEPStreamingBuffer,
        "_state",
        classmethod(lambda cls: state),
    )
    monkeypatch.setattr(deepep_dispatcher_module, "ElasticBuffer", object)
    with pytest.raises(RuntimeError, match="wavefront slots"):
        DeepEPStreamingBuffer.get_buffer(
            FakeGroup(),
            4096,
            8,
            256,
            False,
            wavefront_slot=0,
            num_wavefront_slots=1,
        )


def test_streaming_staged_dispatch_keeps_two_wavefront_slots_independent():
    dispatcher = object.__new__(DeepEPDispatcher)
    dispatcher.streaming_enabled = True
    dispatcher._streaming_dispatch_intermediate = {}
    dispatcher._streaming_combine_intermediate = {}
    dispatcher.dispatch_streaming = lambda hidden, topk, *, wavefront_slot: (
        SimpleNamespace(wavefront_slot=wavefront_slot, hidden=hidden)
    )

    dispatcher.dispatch_a("request-a", "topk-a", tbo_subbatch_index=0)
    dispatcher.dispatch_a("request-b", "topk-b", tbo_subbatch_index=1)
    request_b = dispatcher.dispatch_b(tbo_subbatch_index=1)
    request_a = dispatcher.dispatch_b(tbo_subbatch_index=0)

    assert request_a.hidden == "request-a"
    assert request_b.hidden == "request-b"
    result_a = SimpleNamespace(dispatch=request_a, output="output-a")
    result_b = SimpleNamespace(dispatch=request_b, output="output-b")
    dispatcher.combine_a(result_a, tbo_subbatch_index=0)
    dispatcher.combine_a(result_b, tbo_subbatch_index=1)
    assert dispatcher.combine_b(tbo_subbatch_index=1) == "output-b"
    assert dispatcher.combine_b(tbo_subbatch_index=0) == "output-a"


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


def test_bf16_streaming_forwards_per_expert_shape_hint_to_both_gemms():
    signature = inspect.signature(launch_bf16_streaming_moe)
    assert signature.parameters["expected_m_per_expert"].default is None

    source = inspect.getsource(launch_bf16_streaming_moe)
    assert source.count(
        "expected_m_for_psum_layout=expected_m_per_expert"
    ) == 2
    assert "expected_m_per_expert must be a positive integer" in source
