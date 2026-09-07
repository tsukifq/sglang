import inspect
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.batch_overlap.operations import _resolve_tbo_child_contexts
from sglang.srt.layers.moe.deepep_streaming_kernels import masked_route_weight_mul_
from sglang.srt.layers.moe.deepep_streaming import (
    DeepEPStreamingDispatch,
    _iter_ready_cuda_events,
    _lane_layout_from_psum,
    _launch_streaming_moe_lanes,
    _launch_streaming_moe_waves,
    _rank_merged_input_scale_view,
    _require_per_lane_release,
    _resolve_streaming_wave_size,
    configure_deepep_streaming_environment,
    is_deepep_streaming_batched_return_enabled,
    is_deepep_streaming_rank_merge_enabled,
    is_deepep_v2_sync_baseline_enabled,
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
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(
    est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200"
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


def test_streaming_wave_size_tracks_loaded_deep_gemm_api(monkeypatch):
    def current_api(*args, repeat_weight_groups=False, **kwargs):
        return None

    def stale_api(*args, **kwargs):
        return None

    monkeypatch.delenv("SGLANG_DEEPEP_STREAMING_WAVE_SIZE", raising=False)
    assert (
        _resolve_streaming_wave_size(
            lanes=4, device_major=10, grouped_gemm=current_api
        )
        == 4
    )
    assert (
        _resolve_streaming_wave_size(
            lanes=8, device_major=10, grouped_gemm=current_api
        )
        == 8
    )
    assert (
        _resolve_streaming_wave_size(
            lanes=4, device_major=10, grouped_gemm=stale_api
        )
        == 1
    )

    monkeypatch.setenv("SGLANG_DEEPEP_STREAMING_WAVE_SIZE", "4")
    with pytest.raises(RuntimeError, match="repeat_weight_groups"):
        _resolve_streaming_wave_size(
            lanes=4, device_major=10, grouped_gemm=stale_api
        )


def test_rank_merged_input_scales_are_compact_for_each_wave():
    storage = torch.empty((2, 7, 128), dtype=torch.int32)

    first = _rank_merged_input_scale_view(storage, 0, 128)
    second = _rank_merged_input_scale_view(storage, 1, 128)

    assert first.shape == (128, 7)
    assert first.stride() == (1, 128)
    assert second.shape == (128, 7)
    assert second.stride() == (1, 128)
    assert second.data_ptr() - first.data_ptr() == 7 * 128 * storage.element_size()


def test_rank_merged_input_scales_reject_cross_wave_stride():
    shared = torch.empty((7, 256), dtype=torch.int32)
    cross_wave = shared.narrow(1, 128, 128).unsqueeze(0)
    second_wave = cross_wave[0].t()

    assert second_wave.stride() == (1, 256)
    with pytest.raises(ValueError, match="compact per wave"):
        _rank_merged_input_scale_view(cross_wave, 0, 128)


def test_streaming_wave_size_reads_pybind_doc_when_signature_is_missing(
    monkeypatch,
):
    def pybind_api(*args, **kwargs):
        return None

    pybind_api.__doc__ = (
        "m_grouped(..., repeat_weight_groups: bool = False) -> None"
    )
    real_signature = inspect.signature

    def signature(value):
        if value is pybind_api:
            raise ValueError("no signature found for builtin")
        return real_signature(value)

    monkeypatch.setattr(
        "sglang.srt.layers.moe.deepep_streaming.inspect.signature",
        signature,
    )
    monkeypatch.delenv("SGLANG_DEEPEP_STREAMING_WAVE_SIZE", raising=False)

    assert (
        _resolve_streaming_wave_size(
            lanes=4,
            device_major=10,
            grouped_gemm=pybind_api,
        )
        == 4
    )


def test_ready_event_polling_can_avoid_scheduler_yields(monkeypatch):
    class Event:
        def __init__(self, ready_after):
            self.ready_after = ready_after
            self.queries = 0

        def query(self):
            self.queries += 1
            return self.queries > self.ready_after

    sleeps = []
    monkeypatch.setattr(
        "sglang.srt.layers.moe.deepep_streaming.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    monkeypatch.setenv(
        "SGLANG_DEEPEP_STREAMING_HOST_POLL_YIELD_AFTER", "0"
    )

    assert list(_iter_ready_cuda_events([Event(3), Event(1)])) == [1, 0]
    assert sleeps == []


def test_ready_event_polling_rejects_invalid_backoff(monkeypatch):
    monkeypatch.setenv(
        "SGLANG_DEEPEP_STREAMING_HOST_POLL_YIELD_AFTER", "-1"
    )

    with pytest.raises(ValueError, match="must be nonnegative"):
        list(_iter_ready_cuda_events([]))


def test_ready_event_polling_preserves_default_yield(monkeypatch):
    class Event:
        def __init__(self):
            self.queries = 0

        def query(self):
            self.queries += 1
            return self.queries > 1

    sleeps = []
    monkeypatch.delenv(
        "SGLANG_DEEPEP_STREAMING_HOST_POLL_YIELD_AFTER", raising=False
    )
    monkeypatch.setattr(
        "sglang.srt.layers.moe.deepep_streaming.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    assert list(_iter_ready_cuda_events([Event()])) == [0]
    assert sleeps == [0]


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


def test_v2_sync_baseline_gate_is_explicit(monkeypatch):
    monkeypatch.delenv("SGLANG_DEEPEP_V2_SYNC_BASELINE", raising=False)
    assert not is_deepep_v2_sync_baseline_enabled()
    monkeypatch.setenv("SGLANG_DEEPEP_V2_SYNC_BASELINE", "1")
    assert is_deepep_v2_sync_baseline_enabled()

    source = inspect.getsource(_launch_streaming_moe_lanes)
    all_lane_wait = source.index("for lane in range(lanes):")
    barrier_event = source.index("all_lanes_ready.record(source_stream)")
    lane_compute = source.index("lane_compute(lane, lane_output[lane])")
    assert all_lane_wait < barrier_event < lane_compute


def test_streaming_rank_merge_defaults_on_for_supported_fp4(monkeypatch):
    monkeypatch.delenv("SGLANG_DEEPEP_STREAMING_RANK_MERGE", raising=False)
    assert not is_deepep_streaming_rank_merge_enabled()
    assert is_deepep_streaming_rank_merge_enabled(supported=True, wave_size=4)
    monkeypatch.setenv("SGLANG_DEEPEP_STREAMING_RANK_MERGE", "1")
    assert is_deepep_streaming_rank_merge_enabled()
    monkeypatch.setenv("SGLANG_DEEPEP_STREAMING_RANK_MERGE", "0")
    assert not is_deepep_streaming_rank_merge_enabled(supported=True, wave_size=4)


@pytest.mark.parametrize("wave_size", [1, 2, 4, 8])
@pytest.mark.parametrize("layout_supported", [False, True])
def test_rank_merge_auto_respects_layout_and_resolved_wave(
    monkeypatch, wave_size, layout_supported
):
    monkeypatch.delenv("SGLANG_DEEPEP_STREAMING_RANK_MERGE", raising=False)
    assert is_deepep_streaming_rank_merge_enabled(
        supported=layout_supported, wave_size=wave_size
    ) == (layout_supported and wave_size > 1)


def test_rank_merge_explicit_override_remains_visible_to_validation(monkeypatch):
    # The launcher must still reject an explicitly forced incompatible mode.
    monkeypatch.setenv("SGLANG_DEEPEP_STREAMING_RANK_MERGE", "1")
    assert is_deepep_streaming_rank_merge_enabled(supported=True, wave_size=1)
    monkeypatch.setenv("SGLANG_DEEPEP_STREAMING_RANK_MERGE", "0")
    assert not is_deepep_streaming_rank_merge_enabled(supported=True, wave_size=4)


def test_streaming_batched_return_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SGLANG_DEEPEP_STREAMING_BATCHED_RETURN", raising=False)
    assert not is_deepep_streaming_batched_return_enabled()

    monkeypatch.setenv("SGLANG_DEEPEP_STREAMING_BATCHED_RETURN", "1")
    assert is_deepep_streaming_batched_return_enabled()


def test_streaming_batched_return_precedes_lane_release():
    source = inspect.getsource(_launch_streaming_moe_waves)
    batched_return = source.index("streaming_combine_return_many(")
    lane_release = source.index(
        "release_streaming_lane(lane, dispatch.generation)", batched_return
    )
    batch_finalized = source.index("batch_finalized.record(stream)", lane_release)

    assert batched_return < lane_release < batch_finalized


def test_streaming_dispatch_rejects_incomplete_runtime_view():
    with pytest.raises(ValueError, match="seven DeepEP lane-view fields"):
        DeepEPStreamingDispatch.from_runtime(
            buffer=object(),
            raw=(object(),),
            source_topk_idx=None,
            transport_handle=object(),
            transport_event=object(),
        )


def test_streaming_layer_gates_sm100_transport_before_lane_consumers():
    source = inspect.getsource(_launch_streaming_moe_lanes)
    assert "cuStreamWaitValue64" in source
    assert "streaming_combine_return" in source
    assert "streaming_combine_reduce" in source
    assert "release_streaming_lane(lane, dispatch.generation)" in source
    assert "host_lane_gate" in source
    assert "_iter_ready_cuda_events(lane_ready_events)" in source
    assert ".synchronize(" not in source
    assert ".barrier(" not in source


def test_streaming_lane_ack_is_ordered_after_combine_return_submission():
    source = inspect.getsource(_launch_streaming_moe_lanes)

    pack_wait = source.index("cuStreamWaitValue64")
    release_lane = source.index("release_streaming_lane(lane, dispatch.generation)")
    lane_compute = source.index("lane_compute(lane, lane_output[lane])")
    combine_return = source.index("dispatch.buffer.streaming_combine_return(")

    assert pack_wait < lane_compute < combine_return < release_lane


def test_streaming_generation_owned_control_removes_deferred_psum_copy():
    source = inspect.getsource(_launch_streaming_moe_lanes)

    assert "expert_psum_snapshot" not in source


def test_streaming_wave_profile_marks_shared_compute_events():
    source = inspect.getsource(_launch_streaming_moe_waves)

    assert '"compute_scope": "wave"' in source
    assert '"compute_group_id": wave_index' in source
    assert "streaming_combine_return(" in source
    assert "release_streaming_lane(lane, dispatch.generation)" in source


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
    signature = inspect.signature(launch_fp8_streaming_moe)
    assert signature.parameters["swiglu_limit"].default is None
    assert signature.parameters["is_fp4_expert"].default is False

    source = inspect.getsource(launch_fp8_streaming_moe)
    assert source.count("deep_gemm.m_grouped_fp8_gemm_nt_contiguous(") == 6
    assert source.count("use_psum_layout=True") == 6
    assert "fuse_silu_and_mul=True" in source
    assert "silu_and_mul_clamp(gate_up[lane], down_input[lane], swiglu_limit)" in source
    assert "fuse_silu_and_mul=False" in source
    assert "silu_and_mul_contig_post_quant" not in source
    assert "swiglu_limit=swiglu_limit" in source
    assert '"recipe_a": (1, 128)' in source
    assert '"recipe_b": (1, 32)' in source
    assert "m_indices" not in source
    assert "shadow" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_masked_route_weight_mul_preserves_inactive_tail():
    output = torch.arange(32, dtype=torch.float32, device="cuda").reshape(8, 4)
    original = output.clone()
    route_weights = torch.linspace(0.5, 1.5, 8, device="cuda")
    active_rows = torch.tensor([3], dtype=torch.int32, device="cuda")

    masked_route_weight_mul_(output, route_weights, active_rows)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output[:3], original[:3] * route_weights[:3, None]
    )
    torch.testing.assert_close(output[3:], original[3:])


def test_bf16_streaming_forwards_per_expert_shape_hint_to_both_gemms():
    signature = inspect.signature(launch_bf16_streaming_moe)
    assert signature.parameters["expected_m_per_expert"].default is None
    assert signature.parameters["swiglu_limit"].default is None

    source = inspect.getsource(launch_bf16_streaming_moe)
    assert source.count(
        "expected_m_for_psum_layout=expected_m_per_expert"
    ) == 2
    assert "expected_m_per_expert must be a positive integer" in source
