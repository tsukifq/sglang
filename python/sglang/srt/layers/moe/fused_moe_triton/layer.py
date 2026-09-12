# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

import logging
import os
import time
from enum import Enum
from functools import cached_property
from typing import List, Optional, Tuple

import torch
from torch.nn.parameter import UninitializedParameter

from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.distributed import (
    get_moe_ep_group,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.deepep_streaming import (
    DeepEPStreamingDispatch,
    DeepEPStreamingLayerResult,
    _lane_layout_from_psum,
    is_deepep_streaming_enabled,
    launch_bf16_streaming_moe,
    launch_fp8_streaming_moe,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    create_kt_config_from_server_args,
)
from sglang.srt.layers.moe.profiling import (
    build_moe_component_profile,
    best_cuda_clock_anchor,
    calibrated_event_timing_guard_ns,
    cuda_clock_anchor_attempts,
    cuda_event_host_interval,
    emit_moe_timeline_record,
    ensure_moe_timeline_collector,
    host_clock_domain_id,
    moe_timeline_scope,
    record_moe_timeline_event,
    submit_moe_timeline_collection,
)
from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
    AscendTPDispatcher,
)
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    has_per_rank_fused_shared_slots,
    uses_per_rank_fused_shared_slots,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMxInt4MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import quantize_block_fp8_weight_to_mxfp4
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    print_info_once,
    round_up,
)
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


def _get_deepep_comm_group(a2a_backend):
    group = get_tp_group().device_group

    if a2a_backend.is_mori():
        group = get_tp_group()

    elif _is_npu:
        group = get_moe_ep_group().device_group

    return group


def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
    a2a_backend = get_moe_a2a_backend()
    if is_deepep_streaming_enabled() and not a2a_backend.is_deepep():
        raise ValueError(
            "SGLANG_ENABLE_DEEPEP_STREAMING requires --moe-a2a-backend deepep"
        )
    if a2a_backend.is_none() and is_npu():
        return AscendTPDispatcher(moe_runner_config)
    elif (
        a2a_backend.is_none()
        or a2a_backend.is_megamoe()
        or a2a_backend.is_ascend_fuseep()
    ):
        # ascend_fuseep bypasses the dispatcher abstraction (see
        # forward_fuseep in hardware_backend/npu/moe/fuseep.py); a
        # StandardDispatcher is created but never invoked.
        return StandardDispatcher(moe_runner_config)
    elif (
        a2a_backend.is_deepep()
        or a2a_backend.is_mooncake()
        or a2a_backend.is_mori()
        or a2a_backend.is_nixl()
    ):
        dispatcher_kwargs = dict(
            group=_get_deepep_comm_group(a2a_backend),
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=True,
        )
        if is_deepep_streaming_enabled():
            return DeepEPDispatcher(**dispatcher_kwargs)
        return MaybeTboDeepEPDispatcher(
            **dispatcher_kwargs,
        )
    elif a2a_backend.is_flashinfer():
        return FlashinferDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
        )
    else:
        raise NotImplementedError(f"Unsupported a2a backend: {a2a_backend}")


# DeepEP transport sidecars can outlive the Python call that submitted them.
# Reap them process-wide so a completed generation from an earlier model layer
# is released promptly instead of pinning one lane-capacity allocation per MoE
# layer until the next batch reaches that same layer.
_deepep_streaming_global_inflight: list[DeepEPStreamingLayerResult] = []
_DEEPEP_STREAMING_GENERATION_GRACE = 4


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to apply all_reduce on the output of the layer
        quant_config: Quantization configuration.
        inplace: suggestion to compute inplace (modify input activation).
    """

    # True on shared-expert FusedMoE subclasses (e.g. Inkling's sink); lets
    # backend resolution distinguish them from routed experts.
    is_shared_fused_moe = False

    _skip_aiter_moe_shuffle: bool = False

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        use_weight_loader_fused: bool = False,
        with_bias=False,
        routing_method_type: Optional[RoutingMethodType] = None,
        is_gated: bool = True,
        gate_up_interleaved: bool = True,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.params_dtype = params_dtype
        self.layer_name = prefix
        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.with_bias = with_bias
        self.num_fused_shared_experts = num_fused_shared_experts

        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.moe_ep_size = get_parallel().moe_ep_size
        self.moe_ep_rank = get_parallel().moe_ep_rank
        self.moe_tp_size = get_parallel().moe_tp_size
        self.moe_tp_rank = get_parallel().moe_tp_rank

        # For fused shared experts, DeepEP-class and MegaMOE backends use
        # per-rank physical shared slots, while other backends keep fused
        # shared experts as global shared slots. When fusion is disabled,
        # num_fused_shared_experts is 0 and no shared slots are added here.
        if has_per_rank_fused_shared_slots(num_fused_shared_experts):
            num_shared_slots = num_fused_shared_experts * self.moe_ep_size
        else:
            num_shared_slots = num_fused_shared_experts

        self._num_global_routed = num_experts - num_shared_slots
        server_args = get_server_args()
        if server_args.ep_join_mode == "scale":
            storage_ep_size = server_args.elastic_ep_initial_size
            assert storage_ep_size is not None
            self._expert_storage_rank = (
                server_args.ep_join_rank_offset + self.moe_ep_rank
            )
        else:
            storage_ep_size = self.moe_ep_size
            self._expert_storage_rank = self.moe_ep_rank
        assert self._num_global_routed % storage_ep_size == 0
        self._num_local_routed = self._num_global_routed // storage_ep_size
        self.num_local_experts = self._num_local_routed + num_fused_shared_experts
        self._has_fused_shared = num_fused_shared_experts > 0
        self._pending_fp8_shared_weights: dict[tuple[int, str], torch.Tensor] = {}
        self._pending_fp8_shared_scales: dict[tuple[int, str], torch.Tensor] = {}

        assert intermediate_size % self.moe_tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results
        self.use_presharded_weights = use_presharded_weights

        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernels()

        self.use_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        )
        self.use_deep_gemm = get_moe_runner_backend().is_deep_gemm()

        # flashinfer_trtllm kernel requires intermediate_size to be a multiple of 128
        # Pad the intermediate_size_per_partition if necessary
        if (
            self.use_flashinfer_trtllm_moe
            and self.intermediate_size_per_partition % 128 != 0
        ):
            self.intermediate_size_per_partition = round_up(
                self.intermediate_size_per_partition, 128
            )

        self.quant_config = quant_config
        self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()
        # TODO maybe we should remove this `if`, since `Mxfp4MoEMethod` does another round-up logic
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.use_flashinfer_mxfp4_moe
        ):
            hidden_size = round_up(hidden_size, 256)
        self.hidden_size = hidden_size

        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            swiglu_limit=swiglu_limit,
            is_gated=is_gated,
            routing_method_type=routing_method_type,
            gate_up_interleaved=gate_up_interleaved,
        )

        self.quant_method: Optional[FusedMoEMethodBase] = None
        server_args = get_server_args()
        kt_config = create_kt_config_from_server_args(server_args, layer_id)
        if kt_config is not None:
            if quant_config is not None:
                gpu_method = quant_config.get_quant_method(self, prefix)
            else:
                gpu_method = UnquantizedFusedMoEMethod(self.use_triton_kernels)
            self.quant_method = KTEPWrapperMethod(gpu_method, kt_config)
        else:
            if quant_config is not None:
                self.quant_method = quant_config.get_quant_method(self, prefix)
            if self.quant_method is None:
                self.quant_method = UnquantizedFusedMoEMethod(
                    self.use_triton_kernels,
                    self.use_flashinfer_trtllm_moe,
                    self.use_deep_gemm,
                )
        self.supports_deferred_finalize = (
            envs.SGLANG_ENABLE_MOE_DEFERRED_FINALIZE.get()
            and get_moe_runner_backend().is_flashinfer_trtllm()
            and isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
        )
        print_info_once(
            "FlashInfer TRTLLM MoE deferred finalize is "
            f"{'enabled' if self.supports_deferred_finalize else 'disabled'} "
            f"(moe_runner_backend={server_args.moe_runner_backend}, "
            f"quant_method={type(self.quant_method).__name__})."
        )

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=(
                self.weight_loader
                if not use_weight_loader_fused
                else self.weight_loader_fused
            ),
            with_bias=with_bias,
            moe_intermediate_size=intermediate_size,
        )

        self.quant_method.create_moe_runner(self, self.moe_runner_config)
        self.dispatcher = create_moe_dispatcher(self.moe_runner_config)
        self._deepep_streaming_streams = {}
        self._deepep_streaming_drain_streams = {}
        self._deepep_streaming_activation_streams = {}
        self._deepep_streaming_inflight = {}
        self._deepep_streaming_fp8 = False
        self._deepep_streaming_fp4 = False
        self._deepep_timeline_calls = 0
        self._deepep_timeline_target_call = 0
        self._deepep_timeline_call_stride = 0
        self._deepep_timeline_call_offset = 0
        self._deepep_timeline_detail = "full"
        self._deepep_timeline_enable_file = ""
        self._deepep_timeline_run_id = ""
        self._deepep_timeline_event_guard_ns = 0
        self._deepep_timeline_calibration_id = ""
        self._deepep_timeline_max_event_to_anchor_ns = 0
        self._deepep_timeline_clock_domain = ""
        self._deepep_timeline_layer_weight = 1
        self._deepep_timeline_selected_layer = False
        timeline_enabled = os.getenv("SGLANG_DEEPEP_TIMELINE", "0").lower()
        if timeline_enabled not in ("", "0", "false", "no", "n"):
            try:
                layer_spec = os.getenv("SGLANG_DEEPEP_TIMELINE_LAYERS", "").strip()
                target_layers = (
                    {int(value) for value in layer_spec.split(",") if value.strip()}
                    if layer_spec
                    else {int(os.getenv("SGLANG_DEEPEP_TIMELINE_LAYER", "0"))}
                )
                self._deepep_timeline_target_call = int(
                    os.getenv("SGLANG_DEEPEP_TIMELINE_CALL", "0")
                )
                self._deepep_timeline_call_stride = int(
                    os.getenv("SGLANG_DEEPEP_TIMELINE_CALL_STRIDE", "0")
                )
                self._deepep_timeline_call_offset = int(
                    os.getenv("SGLANG_DEEPEP_TIMELINE_CALL_OFFSET", "0")
                )
                self._deepep_timeline_layer_weight = int(
                    os.getenv("SGLANG_DEEPEP_TIMELINE_LAYER_WEIGHT", "1")
                )
            except ValueError as error:
                raise ValueError(
                    "DeepEP timeline layer/call sampling values must be integers"
                ) from error
            if self._deepep_timeline_call_stride < 0:
                raise ValueError("SGLANG_DEEPEP_TIMELINE_CALL_STRIDE must be nonnegative")
            if self._deepep_timeline_call_stride and not (
                0 <= self._deepep_timeline_call_offset
                < self._deepep_timeline_call_stride
            ):
                raise ValueError(
                    "SGLANG_DEEPEP_TIMELINE_CALL_OFFSET must be within the stride"
                )
            if self._deepep_timeline_layer_weight <= 0:
                raise ValueError("SGLANG_DEEPEP_TIMELINE_LAYER_WEIGHT must be positive")
            self._deepep_timeline_detail = os.getenv(
                "SGLANG_DEEPEP_TIMELINE_DETAIL", "full"
            ).strip().lower()
            if self._deepep_timeline_detail not in ("full", "arrival"):
                raise ValueError(
                    "SGLANG_DEEPEP_TIMELINE_DETAIL must be full or arrival"
                )
            self._deepep_timeline_enable_file = os.getenv(
                "SGLANG_DEEPEP_TIMELINE_ENABLE_FILE", ""
            ).strip()
            self._deepep_timeline_run_id = os.getenv(
                "SGLANG_DEEPEP_TIMELINE_RUN_ID", ""
            ).strip()
            self._deepep_timeline_event_guard_ns = (
                calibrated_event_timing_guard_ns()
            )
            if (
                self._deepep_timeline_detail == "arrival"
                and not self._deepep_timeline_run_id
            ):
                raise ValueError(
                    "SGLANG_DEEPEP_TIMELINE_RUN_ID is required for arrival profiling"
                )
            if self._deepep_timeline_detail == "arrival":
                self._deepep_timeline_calibration_id = os.getenv(
                    "SGLANG_DEEPEP_TIMELINE_CALIBRATION_ID", ""
                ).strip()
                if not self._deepep_timeline_calibration_id:
                    raise ValueError(
                        "SGLANG_DEEPEP_TIMELINE_CALIBRATION_ID is required "
                        "for arrival profiling"
                    )
                try:
                    self._deepep_timeline_max_event_to_anchor_ns = int(
                        os.environ[
                            "SGLANG_DEEPEP_TIMELINE_MAX_EVENT_TO_ANCHOR_NS"
                        ]
                    )
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        "SGLANG_DEEPEP_TIMELINE_MAX_EVENT_TO_ANCHOR_NS must "
                        "be set from the matched clock calibration"
                    ) from error
                if self._deepep_timeline_max_event_to_anchor_ns <= 0:
                    raise ValueError(
                        "SGLANG_DEEPEP_TIMELINE_MAX_EVENT_TO_ANCHOR_NS must "
                        "be positive"
                    )
            self._deepep_timeline_clock_domain = host_clock_domain_id()
            self._deepep_timeline_selected_layer = (
                get_moe_a2a_backend().is_deepep() and self.layer_id in target_layers
            )
            if self._deepep_timeline_selected_layer:
                ensure_moe_timeline_collector()
        if getattr(self.dispatcher, "streaming_enabled", False):
            self._validate_deepep_streaming()
        self._use_ascend_fuseep = get_moe_a2a_backend().is_ascend_fuseep()

        if (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            if self.moe_runner_config.inplace:
                print_info_once(
                    "Setting inplace to False for FlashInfer TRTLLM MoE backend."
                )
            self.moe_runner_config.inplace = False

        self.should_fuse_routed_scaling_factor_in_topk = (
            isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
            or (
                isinstance(self.quant_method, Fp8MoEMethod)
                and (
                    get_moe_runner_backend().is_cutlass()
                    or get_moe_runner_backend().is_flashinfer_trtllm_routed()
                )
            )
            or (
                isinstance(self.quant_method, UnquantizedFusedMoEMethod)
                and get_moe_runner_backend().is_flashinfer_trtllm_routed()
            )
        )

        self.routing_method_type = routing_method_type

        # overlap args
        self.down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        if self.quant_method is not None and hasattr(self.quant_method, "runner"):
            self.runner = self.quant_method.runner

    @cached_property
    def use_padded_loading(self) -> bool:
        # This handles the case where the loaded weights are smaller than the padded expert_data
        # Use narrow_padded_param_and_loaded_weight for:
        # 1. CPU (always)
        # 2. GPU with flashinfer_trtllm padding (when intermediate_size is padded to 128)
        # 3. GPU with Aiter padding
        aiter_padded = (
            _use_aiter
            and hasattr(self, "w2_weight")
            and getattr(self.w2_weight, "weight_padded", False)
        )

        return _is_cpu or self.use_flashinfer_trtllm_moe or aiter_padded

    def _validate_deepep_streaming(self) -> None:
        """Fail closed for model shapes not covered by the EP4/EP8 milestone."""

        unsupported = []
        if self.moe_ep_size not in (4, 8) or self.moe_tp_size != 1:
            unsupported.append("EP4 or EP8 with MoE TP1")
        if not self.use_deep_gemm:
            unsupported.append("--moe-runner-backend deep_gemm")
        if self.quant_config is None:
            if (
                self.w13_weight.dtype != torch.bfloat16
                or self.w2_weight.dtype != torch.bfloat16
            ):
                unsupported.append("BF16 W13/W2 weights")
        elif isinstance(self.quant_method, Fp8MoEMethod):
            self._deepep_streaming_fp8 = True
            self._deepep_streaming_fp4 = self.quant_method.is_fp4_expert
            if (
                not self.quant_method.block_quant
                or tuple(self.quant_method.weight_block_size or ()) != (128, 128)
                or self.quant_method.use_mxfp8
            ):
                unsupported.append(
                    "dynamic block-FP8 or DSV4 MXFP4 experts"
                )
            if self._deepep_streaming_fp4:
                if (
                    self.w13_weight.dtype != torch.int8
                    or self.w2_weight.dtype != torch.int8
                ):
                    unsupported.append("packed int8 DSV4 MXFP4 W13/W2 weights")
            elif (
                self.w13_weight.dtype != torch.float8_e4m3fn
                or self.w2_weight.dtype != torch.float8_e4m3fn
            ):
                unsupported.append("e4m3 W13/W2 weights")
            if (
                self.w13_weight_scale_inv.dtype != torch.float32
                or self.w2_weight_scale_inv.dtype != torch.float32
            ):
                unsupported.append("float32 pre-load expert weight scales")
            if self.quant_method.quant_config.activation_scheme != "dynamic":
                unsupported.append("dynamic FP8 activation scaling")
        else:
            unsupported.append(
                "BF16, block-FP8, or DSV4 MXFP4 expert quantization"
            )
        if self.num_fused_shared_experts != 0:
            unsupported.append("shared-expert fusion disabled")
        if self.reduce_results:
            unsupported.append("reduce_results=False")
        if self.with_bias:
            unsupported.append("bias-free experts")
        if self.moe_runner_config.no_combine:
            unsupported.append("no_combine=False")
        if (
            not self.moe_runner_config.is_gated
            or self.moe_runner_config.activation != "silu"
        ):
            unsupported.append("gated SiLU experts")
        if any(
            value is not None
            for value in (
                self.moe_runner_config.gemm1_alpha,
                self.moe_runner_config.gemm1_clamp_limit,
            )
        ):
            unsupported.append("no GEMM1 alpha or clamp variant")
        if unsupported:
            raise ValueError(
                "SGLANG_ENABLE_DEEPEP_STREAMING currently requires: "
                + ", ".join(unsupported)
            )

    def _deepep_timeline_context(self, mode: str) -> Optional[dict]:
        """Select one DeepEP layer invocation for a deferred GPU timeline."""

        if not self._deepep_timeline_selected_layer:
            return None
        call_index = self._deepep_timeline_calls
        self._deepep_timeline_calls += 1
        if self._deepep_timeline_enable_file and not os.path.exists(
            self._deepep_timeline_enable_file
        ):
            return None
        if self._deepep_timeline_call_stride:
            if (
                call_index % self._deepep_timeline_call_stride
                != self._deepep_timeline_call_offset
            ):
                return None
        elif call_index != self._deepep_timeline_target_call:
            return None

        def enabled(name: str) -> bool:
            return os.getenv(name, "0").lower() not in (
                "",
                "0",
                "false",
                "no",
                "n",
            )

        if mode == "streaming":
            execution_variant = (
                "streaming-v2-arrival-gated"
                if enabled("SGLANG_DEEPEP_V2_SYNC_BASELINE")
                else "streaming-v2-async"
            )
        elif enabled("SGLANG_DEEPEP_RANK_READY"):
            execution_variant = "staged-rank-ready"
        elif enabled("SGLANG_DEEPEP_V2_BASELINE"):
            execution_variant = "staged-v2-arrival-gated"
        else:
            execution_variant = "staged-public"
        return {
            "rank": self.moe_ep_rank,
            "layer_id": self.layer_id,
            "call_index": call_index,
            "mode": mode,
            "execution_variant": execution_variant,
            "profile_detail": self._deepep_timeline_detail,
            "run_id": self._deepep_timeline_run_id or None,
            "sample_id": (
                f"{self._deepep_timeline_run_id}:{execution_variant}:"
                f"layer={self.layer_id}:call={call_index}"
            ),
            "host_clock_domain_id": self._deepep_timeline_clock_domain,
            "sampling": {
                "call_stride": self._deepep_timeline_call_stride or 1,
                "call_offset": self._deepep_timeline_call_offset,
                "layer_weight": self._deepep_timeline_layer_weight,
            },
            "clock_contract": {
                "event_timing_guard_ns": self._deepep_timeline_event_guard_ns,
                "calibration_id": self._deepep_timeline_calibration_id,
                "max_event_to_anchor_ns": (
                    self._deepep_timeline_max_event_to_anchor_ns
                ),
                "anchor_attempts": cuda_clock_anchor_attempts(),
            },
        }

    @staticmethod
    def _emit_arrival_timeline(
        context: dict,
        origin: torch.cuda.Event,
        output_ready: torch.cuda.Event,
        hidden_states: torch.Tensor,
        timeline: dict,
    ) -> None:
        """Queue a minimal cross-rank MoE-entry sample off the serving thread."""

        defer_started_ns = time.monotonic_ns()
        device = hidden_states.device
        input_tokens = hidden_states.size(0)
        events = dict(timeline["events"])
        recorded = frozenset(timeline["recorded"])
        context = dict(context)
        defer_state = {"done_ns": None}

        def collect() -> None:
            collector_started_ns = time.monotonic_ns()
            with torch.cuda.device(device):
                output_ready.synchronize()
                output_wait_done_ns = time.monotonic_ns()
                anchor_selection = best_cuda_clock_anchor(torch.cuda, device)
                profile_stream = anchor_selection["stream"]
                clock_anchor = anchor_selection["event"]
                anchor_bracket_start_ns = anchor_selection["bracket_start_ns"]
                anchor_bracket_end_ns = anchor_selection["bracket_end_ns"]

            anchor_midpoint_ns = (
                anchor_bracket_start_ns + anchor_bracket_end_ns
            ) // 2

            def aligned_timestamp(event: torch.cuda.Event) -> dict:
                return {
                    "rank_local_ms": origin.elapsed_time(event),
                    **cuda_event_host_interval(
                        event,
                        clock_anchor,
                        anchor_bracket_start_ns=anchor_bracket_start_ns,
                        anchor_bracket_end_ns=anchor_bracket_end_ns,
                        event_timing_guard_ns=int(
                            context["clock_contract"]["event_timing_guard_ns"]
                        ),
                    ),
                }

            collector_before_log_ns = time.monotonic_ns()
            serving_done_ns = defer_state["done_ns"]
            arrival_timestamps = {
                "moe_entry": aligned_timestamp(origin),
                "output_ready": aligned_timestamp(output_ready),
            }
            arrival_timestamps.update(
                {name: aligned_timestamp(events[name]) for name in sorted(recorded)}
            )
            component_events = {
                "layer_entry": arrival_timestamps["moe_entry"],
                "layer_output_ready": arrival_timestamps["output_ready"],
            }
            component_provenance = {
                "layer_entry": (
                    "sglang_caller_stream_event_recorded_at_moe_python_entry"
                ),
                "layer_output_ready": "sglang_caller_stream",
            }
            if "dispatch_input_ready" in recorded:
                component_events["dispatch_input_ready"] = arrival_timestamps[
                    "dispatch_input_ready"
                ]
                component_provenance["dispatch_input_ready"] = (
                    "deepep_comm_stream_after_input_dependency_wait"
                )
            component_profile = build_moe_component_profile(
                execution_model="staged",
                detail=context["profile_detail"],
                events=component_events,
                event_provenance=component_provenance,
                counters={"input_tokens": input_tokens},
                capabilities={
                    "layer_entry_includes_host_launch_arrival": True,
                    "exact_dispatch_input_ready": (
                        "dispatch_input_ready" in recorded
                    ),
                    "exact_dispatch_output_ready": False,
                },
            )
            payload = {
                "schema": "sglang-deepep-arrival-timeline-v2",
                **context,
                "input_tokens": input_tokens,
                "profiler_overhead": {
                    "serving_thread_synchronized": False,
                    "serving_thread_defer_us": (
                        (serving_done_ns - defer_started_ns) / 1e3
                        if serving_done_ns is not None
                        else None
                    ),
                    "collector_queue_delay_us": (
                        collector_started_ns - defer_started_ns
                    )
                    / 1e3,
                    "collector_wait_for_output_ms": (
                        output_wait_done_ns - collector_started_ns
                    )
                    / 1e6,
                    "collector_before_log_ms": (
                        collector_before_log_ns - collector_started_ns
                    )
                    / 1e6,
                    "metadata_d2h_bytes": 0,
                    "timed_cuda_event_count": len(recorded) + 1,
                },
                "clock_alignment": {
                    "method": (
                        "minimum-width repeated private-stream CUDA anchor "
                        "projected to host CLOCK_MONOTONIC"
                    ),
                    "anchor_host_monotonic_ns_midpoint": anchor_midpoint_ns,
                    "anchor_bracket_start_ns": anchor_bracket_start_ns,
                    "anchor_bracket_end_ns": anchor_bracket_end_ns,
                    "anchor_attempts": anchor_selection["attempts"],
                    "anchor_selected_attempt": anchor_selection[
                        "selected_attempt"
                    ],
                    "anchor_bracket_widths_ns": anchor_selection[
                        "bracket_widths_ns"
                    ],
                    "uncertainty_ns": (
                        anchor_bracket_end_ns - anchor_bracket_start_ns + 1
                    )
                    // 2
                    + int(context["clock_contract"]["event_timing_guard_ns"]),
                    "event_timing_guard_ns": int(
                        context["clock_contract"]["event_timing_guard_ns"]
                    ),
                },
                "arrival_timestamps": arrival_timestamps,
                "component_profile": component_profile,
            }
            emit_moe_timeline_record("DEEPEP_ARRIVAL_TIMELINE", payload)

        submit_moe_timeline_collection(collect)
        defer_state["done_ns"] = time.monotonic_ns()

    @staticmethod
    def _baseline_logical_outbound_dispatch_from_host(
        topk_ids: torch.Tensor,
        moe_ep_size: int,
        num_local_experts: int,
        hidden_bytes: int,
        route_metadata_bytes: int,
    ) -> list[dict]:
        """Count logical payload on a deferred host snapshot."""

        destinations = []
        for destination in range(moe_ep_size):
            lower = destination * num_local_experts
            upper = lower + num_local_experts
            routed = (topk_ids >= lower) & (topk_ids < upper)
            unique_tokens = int(routed.any(dim=1).sum().item())
            routes = int(routed.sum().item())
            destinations.append(
                {
                    "destination_rank": destination,
                    "unique_tokens": unique_tokens,
                    "routes": routes,
                    "logical_payload_bytes": unique_tokens
                    * (hidden_bytes + route_metadata_bytes),
                }
            )
        return destinations

    def _emit_baseline_timeline(
        self,
        context: dict,
        origin: torch.cuda.Event,
        timeline: dict,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        dispatch_output: DispatchOutput,
        dispatch_buffer_rows: int,
    ) -> None:
        """Queue one sample for collection outside the serving thread."""

        defer_started_ns = time.monotonic_ns()
        events = dict(timeline["events"])
        recorded = frozenset(timeline["recorded"])
        timeline_counters = dict(timeline.get("counters", {}))
        output_ready = events["output_ready"]
        device = hidden_states.device
        input_tokens = hidden_states.size(0)
        runner_backend = get_moe_runner_backend().value
        runner_layout = dispatch_output.num_recv_tokens_per_expert
        runner_psum = (
            runner_layout.detach()
            if isinstance(runner_layout, torch.Tensor)
            else None
        )
        runner_rows = (
            []
            if runner_psum is not None
            else [int(value) for value in runner_layout]
        )
        topk_ids = (
            topk_output.topk_ids.detach()
            if TopKOutputChecker.format_is_standard(topk_output)
            else None
        )
        hidden_bytes = hidden_states.size(1) * hidden_states.element_size()
        route_metadata_bytes = (
            topk_output.topk_ids.size(1)
            * (
                topk_output.topk_ids.element_size()
                + topk_output.topk_weights.element_size()
            )
            if topk_ids is not None
            else 0
        )
        moe_ep_size = self.moe_ep_size
        num_local_experts = self.num_local_experts
        context = dict(context)
        defer_state = {"done_ns": None}

        def collect() -> None:
            collector_started_ns = time.monotonic_ns()
            with torch.cuda.device(device):
                output_ready.synchronize()
                output_wait_done_ns = time.monotonic_ns()

                # The serving stream is never synchronized. Once its output
                # event is complete, calibrate on a private idle stream so
                # later serving work cannot move the host-clock anchor.
                anchor_selection = best_cuda_clock_anchor(torch.cuda, device)
                profile_stream = anchor_selection["stream"]
                clock_anchor = anchor_selection["event"]
                anchor_bracket_start_ns = anchor_selection["bracket_start_ns"]
                anchor_bracket_end_ns = anchor_selection["bracket_end_ns"]

                topk_ids_host = None
                runner_psum_host = None
                metadata_d2h_bytes = 0
                if topk_ids is not None:
                    topk_ids_host = torch.empty(
                        topk_ids.shape,
                        dtype=topk_ids.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    metadata_done = torch.cuda.Event()
                    with torch.cuda.stream(profile_stream):
                        topk_ids_host.copy_(topk_ids, non_blocking=True)
                        if runner_psum is not None:
                            runner_psum_host = torch.empty(
                                runner_psum.shape,
                                dtype=runner_psum.dtype,
                                device="cpu",
                                pin_memory=True,
                            )
                            runner_psum_host.copy_(
                                runner_psum, non_blocking=True
                            )
                        metadata_done.record(profile_stream)
                    metadata_done.synchronize()
                    metadata_d2h_bytes = (
                        topk_ids_host.numel() * topk_ids_host.element_size()
                    )
                    if runner_psum_host is not None:
                        metadata_d2h_bytes += (
                            runner_psum_host.numel()
                            * runner_psum_host.element_size()
                        )

            anchor_midpoint_ns = (anchor_bracket_start_ns + anchor_bracket_end_ns) // 2

            def aligned_timestamp(event: torch.cuda.Event) -> dict:
                rank_local_ms = origin.elapsed_time(event)
                return {
                    "rank_local_ms": rank_local_ms,
                    **cuda_event_host_interval(
                        event,
                        clock_anchor,
                        anchor_bracket_start_ns=anchor_bracket_start_ns,
                        anchor_bracket_end_ns=anchor_bracket_end_ns,
                        event_timing_guard_ns=int(
                            context["clock_contract"]["event_timing_guard_ns"]
                        ),
                    ),
                }

            arrival_timestamps = {"moe_entry": aligned_timestamp(origin)}
            arrival_timestamps.update(
                {name: aligned_timestamp(events[name]) for name in sorted(recorded)}
            )
            outbound = (
                self._baseline_logical_outbound_dispatch_from_host(
                    topk_ids_host,
                    moe_ep_size,
                    num_local_experts,
                    hidden_bytes,
                    route_metadata_bytes,
                )
                if topk_ids_host is not None
                else []
            )
            dispatch_done = events["dispatch_done"]
            gemm_done = events["gemm_done"]
            combine_done = events["combine_done"]
            dispatch_ms = origin.elapsed_time(dispatch_done)
            logical_bytes = sum(item["logical_payload_bytes"] for item in outbound)
            profile_runner_rows = runner_rows
            if runner_psum_host is not None:
                profile_runner_rows = _lane_layout_from_psum(
                    runner_psum_host.to(dtype=torch.int64).tolist()
                )["expert_rows"]

            required_detail = {
                "dispatch_prepare_done",
                "dispatch_done",
                "runner_pre_permute_done",
                "w13_start",
                "w13_done",
                "activation_start",
                "activation_done",
                "w2_start",
                "w2_done",
                "runner_post_permute_done",
                "gemm_done",
                "combine_prepare_done",
                "combine_done",
                "output_ready",
            }
            detail_available = required_detail.issubset(recorded)
            breakdown = {
                "available": detail_available,
                "recorded_events": sorted(recorded),
            }
            if detail_available:

                def interval(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
                    return start.elapsed_time(end)

                stages = {
                    "dispatch_prepare_ms": interval(
                        origin, events["dispatch_prepare_done"]
                    ),
                    "dispatch_transport_wait_ms": interval(
                        events["dispatch_prepare_done"], dispatch_done
                    ),
                    "runner_pre_permute_ms": interval(
                        dispatch_done, events["runner_pre_permute_done"]
                    ),
                    "w13_launch_gap_ms": interval(
                        events["runner_pre_permute_done"], events["w13_start"]
                    ),
                    "w13_ms": interval(
                        events["w13_start"], events["w13_done"]
                    ),
                    "activation_launch_gap_ms": interval(
                        events["w13_done"], events["activation_start"]
                    ),
                    "activation_ms": interval(
                        events["activation_start"], events["activation_done"]
                    ),
                    "w2_launch_gap_ms": interval(
                        events["activation_done"], events["w2_start"]
                    ),
                    "w2_ms": interval(
                        events["w2_start"], events["w2_done"]
                    ),
                    "runner_post_permute_ms": interval(
                        events["w2_done"], events["runner_post_permute_done"]
                    ),
                    "runner_finalize_ms": interval(
                        events["runner_post_permute_done"], gemm_done
                    ),
                    "combine_prepare_ms": interval(
                        gemm_done, events["combine_prepare_done"]
                    ),
                    "combine_transport_wait_ms": interval(
                        events["combine_prepare_done"], combine_done
                    ),
                    "output_finalize_ms": interval(combine_done, output_ready),
                }
                breakdown.update(
                    {
                        "stages": stages,
                        "stage_sum_ms": sum(stages.values()),
                        "event_points_ms": {
                            name: origin.elapsed_time(events[name])
                            for name in sorted(required_detail)
                        },
                    }
                )
            deep_gemm_profile = None
            deep_gemm_work = timeline_counters.get("deep_gemm")
            if deep_gemm_work is not None and {
                "w13_start",
                "w13_done",
                "w2_start",
                "w2_done",
            }.issubset(recorded):
                deep_gemm_profile = dict(deep_gemm_work)
                w13_ms = events["w13_start"].elapsed_time(events["w13_done"])
                w2_ms = events["w2_start"].elapsed_time(events["w2_done"])
                gemm_ms = w13_ms + w2_ms
                deep_gemm_profile["measured"] = {
                    "w13_ms": w13_ms,
                    "w13_useful_tflops": (
                        deep_gemm_work["w13"]["useful_flops"] / (w13_ms * 1e9)
                        if w13_ms > 0
                        else None
                    ),
                    "w2_ms": w2_ms,
                    "w2_useful_tflops": (
                        deep_gemm_work["w2"]["useful_flops"] / (w2_ms * 1e9)
                        if w2_ms > 0
                        else None
                    ),
                    "gemm_ms": gemm_ms,
                    "gemm_useful_tflops": (
                        deep_gemm_work["useful_flops"] / (gemm_ms * 1e9)
                        if gemm_ms > 0
                        else None
                    ),
                }
            component_events = {
                "layer_entry": arrival_timestamps["moe_entry"],
                # dispatch_done executes on the consumer stream after
                # DeepEP's completion dependency has been satisfied. In
                # the staged adapter first/all readiness collapse.
                "dispatch_first_output_ready": arrival_timestamps[
                    "dispatch_done"
                ],
                "dispatch_all_output_ready": arrival_timestamps[
                    "dispatch_done"
                ],
                "compute_first_start": arrival_timestamps["dispatch_done"],
                "compute_all_done": arrival_timestamps["gemm_done"],
                "combine_first_start": arrival_timestamps["gemm_done"],
                "combine_all_done": arrival_timestamps["combine_done"],
                "layer_output_ready": arrival_timestamps["output_ready"],
            }
            component_provenance = {
                "layer_entry": (
                    "sglang_caller_stream_event_recorded_at_moe_python_entry"
                ),
                "dispatch_first_output_ready": (
                    "consumer_stream_after_deepep_completion_wait"
                ),
                "dispatch_all_output_ready": (
                    "consumer_stream_after_deepep_completion_wait"
                ),
                "compute_first_start": "sglang_caller_stream_boundary",
                "compute_all_done": "sglang_caller_stream_boundary",
                "combine_first_start": "sglang_caller_stream_boundary",
                "combine_all_done": (
                    "consumer_stream_after_deepep_completion_wait"
                ),
                "layer_output_ready": "sglang_caller_stream",
            }
            if "dispatch_input_ready" in recorded:
                component_events["dispatch_input_ready"] = arrival_timestamps[
                    "dispatch_input_ready"
                ]
                component_provenance["dispatch_input_ready"] = (
                    "deepep_comm_stream_after_input_dependency_wait"
                )
            component_profile = build_moe_component_profile(
                execution_model="staged",
                detail=context["profile_detail"],
                events=component_events,
                event_provenance=component_provenance,
                counters={
                    "input_tokens": input_tokens,
                    "dispatch": {
                        "logical_outbound_payload_bytes": logical_bytes,
                        "received_buffer_rows": dispatch_buffer_rows,
                    },
                    "compute": {
                        "runner_rows": sum(profile_runner_rows),
                        "nonempty_experts": sum(row > 0 for row in profile_runner_rows),
                        "deep_gemm": deep_gemm_profile,
                    },
                },
                items=(
                    [
                        {
                            "kind": "grouped_gemm",
                            "id": name,
                            "events": {
                                "start": arrival_timestamps[f"{name}_start"],
                                "done": arrival_timestamps[f"{name}_done"],
                            },
                            "counters": deep_gemm_work[name],
                        }
                        for name in ("w13", "w2")
                    ]
                    if deep_gemm_profile is not None
                    else []
                )
                + [
                    {
                        "kind": "rank_batch",
                        "id": str(context["rank"]),
                        "events": {
                            "dispatch_output_ready_ms": dispatch_ms,
                            "compute_done_ms": origin.elapsed_time(gemm_done),
                            "combine_done_ms": origin.elapsed_time(combine_done),
                        },
                        "counters": {
                            "runner_rows": sum(profile_runner_rows),
                            "nonempty_experts": sum(
                                row > 0 for row in profile_runner_rows
                            ),
                        },
                    }
                ],
                capabilities={
                    "layer_entry_includes_host_launch_arrival": True,
                    "exact_dispatch_input_ready": (
                        "dispatch_input_ready" in recorded
                    ),
                    "exact_dispatch_output_ready": True,
                },
            )
            collector_before_log_ns = time.monotonic_ns()
            serving_done_ns = defer_state["done_ns"]
            payload = {
                "schema": "sglang-deepep-baseline-timeline-v4",
                **context,
                "input_tokens": input_tokens,
                "runner_backend": runner_backend,
                "profiler_overhead": {
                    "serving_thread_synchronized": False,
                    "serving_thread_defer_us": (
                        (serving_done_ns - defer_started_ns) / 1e3
                        if serving_done_ns is not None
                        else None
                    ),
                    "collector_queue_delay_us": (
                        collector_started_ns - defer_started_ns
                    )
                    / 1e3,
                    "collector_wait_for_output_ms": (
                        output_wait_done_ns - collector_started_ns
                    )
                    / 1e6,
                    "collector_before_log_ms": (
                        collector_before_log_ns - collector_started_ns
                    )
                    / 1e6,
                    "metadata_d2h_bytes": metadata_d2h_bytes,
                    "timed_cuda_event_count": len(recorded) + 1,
                },
                "clock_alignment": {
                    "method": (
                        "minimum-width repeated private-stream CUDA anchor "
                        "projected to host CLOCK_MONOTONIC"
                    ),
                    "anchor_host_monotonic_ns_midpoint": anchor_midpoint_ns,
                    "anchor_bracket_start_ns": anchor_bracket_start_ns,
                    "anchor_bracket_end_ns": anchor_bracket_end_ns,
                    "anchor_attempts": anchor_selection["attempts"],
                    "anchor_selected_attempt": anchor_selection[
                        "selected_attempt"
                    ],
                    "anchor_bracket_widths_ns": anchor_selection[
                        "bracket_widths_ns"
                    ],
                    "uncertainty_ns": (
                        anchor_bracket_end_ns - anchor_bracket_start_ns + 1
                    )
                    // 2
                    + int(context["clock_contract"]["event_timing_guard_ns"]),
                    "event_timing_guard_ns": int(
                        context["clock_contract"]["event_timing_guard_ns"]
                    ),
                },
                "arrival_timestamps": arrival_timestamps,
                "dispatch": {
                    "done_ms": dispatch_ms,
                    "elapsed_ms": dispatch_ms,
                    "logical_outbound_payload_bytes": logical_bytes,
                    "logical_outbound_gbps": logical_bytes / (dispatch_ms * 1e6),
                    "destinations": outbound,
                    "received_buffer_rows_before_runner": dispatch_buffer_rows,
                    "runner_rows": sum(profile_runner_rows),
                    "runner_nonempty_experts": sum(
                        row > 0 for row in profile_runner_rows
                    ),
                    "runner_max_expert_rows": max(profile_runner_rows, default=0),
                    "runner_rows_per_expert": profile_runner_rows,
                },
                "grouped_mlp": {
                    "start_ms": origin.elapsed_time(dispatch_done),
                    "done_ms": origin.elapsed_time(gemm_done),
                    "elapsed_ms": dispatch_done.elapsed_time(gemm_done),
                },
                "combine": {
                    "start_ms": origin.elapsed_time(gemm_done),
                    "done_ms": origin.elapsed_time(combine_done),
                    "elapsed_ms": gemm_done.elapsed_time(combine_done),
                },
                "output_finalize_ms": combine_done.elapsed_time(output_ready),
                "total_ms": origin.elapsed_time(output_ready),
                "breakdown": breakdown,
                "component_profile": component_profile,
            }
            emit_moe_timeline_record("DEEPEP_BASELINE_TIMELINE", payload)

        submit_moe_timeline_collection(collect)
        defer_state["done_ns"] = time.monotonic_ns()

    def _forward_deepep_streaming(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        timeline_context: Optional[dict] = None,
    ) -> torch.Tensor:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError("streaming DeepEP requires standard top-k output")
        if torch.is_grad_enabled():
            raise RuntimeError("streaming DeepEP is inference-only")
        global _deepep_streaming_global_inflight
        # The shared ElasticBuffer's next dispatch consumes generation-1 reuse
        # state. Keep a short completed-generation grace window so transport
        # handles outlive that device-side protocol step; older generations are
        # released as soon as their GPU lifetime fence is complete.
        grace_start = max(
            0,
            len(_deepep_streaming_global_inflight)
            - _DEEPEP_STREAMING_GENERATION_GRACE,
        )
        _deepep_streaming_global_inflight[:] = [
            pending
            for index, pending in enumerate(_deepep_streaming_global_inflight)
            if index >= grace_start or not pending.epoch_drained.query()
        ]
        wavefront_slot = 0
        timeline_origin = None
        dispatch_timeline = None
        if timeline_context is not None:
            timeline_origin = torch.cuda.Event(enable_timing=True)
            timeline_origin.record(torch.cuda.current_stream(hidden_states.device))
            dispatch_timeline = {
                "events": {
                    "dispatch_input_ready": torch.cuda.Event(enable_timing=True)
                },
                "recorded": set(),
                "counters": {},
            }

        resident = None
        if os.getenv("ASYNC_MOE_ONLINE_RESIDENT_MODE", "").lower() not in (
            "", "0", "false", "no", "n"
        ):
            from profiler.online_resident import prepare as prepare_resident

            resident = prepare_resident(self, topk_output.topk_ids)
        if resident is not None and resident.split_group is not None:
            return self._run_deepep_resident_split(
                hidden_states,
                topk_output,
                resident,
                timeline_context=timeline_context,
                timeline_origin=timeline_origin,
                dispatch_timeline=dispatch_timeline,
            )
        with moe_timeline_scope(dispatch_timeline):
            dispatch = self.dispatcher.dispatch_streaming(
                hidden_states=hidden_states,
                topk_output=topk_output,
                wavefront_slot=wavefront_slot,
                physical_topk_ids=(resident.topk_ids if resident is not None else None),
                physical_num_experts=(
                    resident.physical_num_experts if resident is not None else None
                ),
            )
        return self._run_deepep_streaming_dispatch(
            dispatch,
            timeline_context=timeline_context,
            timeline_origin=timeline_origin,
            resident_weights=(resident.weights if resident is not None else None),
        ).output

    def _run_deepep_resident_split(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        resident,
        *,
        timeline_context: Optional[dict],
        timeline_origin: Optional[torch.cuda.Event],
        dispatch_timeline: Optional[dict],
    ) -> torch.Tensor:
        """Prequeue one early-source group and one residual-source group."""

        if os.getenv("ASYNC_MOE_PREPARED_SERVICE", "0") != "1" or os.getenv(
            "ASYNC_MOE_SERVICE_BINDING", "0"
        ) != "1":
            raise RuntimeError(
                "resident split requires the prepared bound service path"
            )
        if self.dispatcher.streaming_num_wavefront_slots != 2:
            raise RuntimeError("resident split requires two ElasticBuffer slots")
        source_streams = getattr(self, "_async_moe_split_source_streams", None)
        if source_streams is None:
            source_streams = tuple(torch.cuda.Stream(priority=0) for _ in range(2))
            self._async_moe_split_source_streams = source_streams
        input_ready = torch.cuda.Event()
        input_ready.record(torch.cuda.current_stream(hidden_states.device))
        results = []
        for group_index, source_stream in enumerate(source_streams):
            source_active = resident.split_group == group_index
            with torch.cuda.stream(source_stream):
                if source_active:
                    source_stream.wait_event(input_ready)
                group_timeline = (
                    None
                    if dispatch_timeline is None
                    else {
                        "events": dict(dispatch_timeline["events"]),
                        "recorded": set(),
                        "counters": {
                            **dispatch_timeline.get("counters", {}),
                            "resident_split_group": group_index,
                        },
                    }
                )
                with moe_timeline_scope(group_timeline):
                    dispatch = self.dispatcher.dispatch_streaming(
                        hidden_states=hidden_states,
                        topk_output=topk_output,
                        wavefront_slot=group_index,
                        physical_topk_ids=resident.topk_ids,
                        physical_num_experts=resident.physical_num_experts,
                        source_active=source_active,
                    )
                group_context = (
                    None
                    if timeline_context is None
                    else {**timeline_context, "resident_split_group": group_index}
                )
                results.append(
                    self._run_deepep_streaming_dispatch(
                        dispatch,
                        timeline_context=group_context,
                        timeline_origin=timeline_origin,
                        resident_weights=resident.weights,
                        resident_binding_variant=f"{resident.binding_prefix}{group_index}",
                    )
                )
        current_stream = torch.cuda.current_stream(hidden_states.device)
        owned_result = results[resident.split_group]
        local_join = os.getenv(
            "ASYNC_MOE_ONLINE_RESIDENT_LOCAL_JOIN", "1"
        ).strip().lower()
        if local_join not in ("0", "1", "false", "true", "no", "yes"):
            raise ValueError(
                "ASYNC_MOE_ONLINE_RESIDENT_LOCAL_JOIN must be boolean"
            )
        joined_results = (
            (owned_result,)
            if local_join in ("1", "true", "yes")
            else tuple(results)
        )
        with torch.cuda.stream(current_stream):
            # Both groups have already been submitted in identical order on
            # every rank. The local-join experiment lets early ranks proceed;
            # the full-join control isolates that scheduling effect while
            # retaining the same two dispatches, weights, and split gate.
            for result in joined_results:
                result.source_ready.current_stream_wait()
        return owned_result.output

    def _run_deepep_streaming_dispatch(
        self,
        dispatch: DeepEPStreamingDispatch,
        *,
        timeline_context: Optional[dict] = None,
        timeline_origin: Optional[torch.cuda.Event] = None,
        resident_weights: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        resident_binding_variant: Optional[str] = None,
    ) -> DeepEPStreamingLayerResult:
        """Consume one request slot without joining another slot's resources."""

        wavefront_slot = dispatch.wavefront_slot
        if wavefront_slot not in self._deepep_streaming_streams:
            self._deepep_streaming_streams[wavefront_slot] = tuple(
                torch.cuda.Stream(priority=0) for _ in range(self.moe_ep_size)
            )
            self._deepep_streaming_drain_streams[wavefront_slot] = (
                torch.cuda.Stream(priority=0)
            )
            self._deepep_streaming_activation_streams[wavefront_slot] = (
                torch.cuda.Stream(priority=0)
            )
        lane_streams = self._deepep_streaming_streams[wavefront_slot]
        drain_stream = self._deepep_streaming_drain_streams[wavefront_slot]
        activation_stream = self._deepep_streaming_activation_streams[wavefront_slot]
        if self._deepep_streaming_fp8:
            fp8_launcher = launch_fp8_streaming_moe
            prepared_service = os.getenv("ASYNC_MOE_PREPARED_SERVICE", "0") == "1"
            if prepared_service:
                # Explicit experiment adapter; bounded cross-layer scratch pool.
                from profiler.prepared_service import get_launcher

                fp8_launcher = get_launcher(fp8_launcher)
            if resident_weights is None:
                w13_weight = self.w13_weight
                w2_weight = self.w2_weight
                w13_scale = self.w13_weight_scale_inv
                w2_scale = self.w2_weight_scale_inv
            else:
                w13_weight, w2_weight, w13_scale, w2_scale = resident_weights
            binding_kwargs = (
                {"_async_moe_binding_variant": resident_binding_variant}
                if prepared_service and resident_binding_variant is not None
                else {}
            )
            result = fp8_launcher(
                dispatch,
                w13_weight,
                w2_weight,
                w13_scale,
                w2_scale,
                self.quant_method.weight_block_size,
                is_fp4_expert=self._deepep_streaming_fp4,
                streams=lane_streams,
                drain_stream=drain_stream,
                activation_stream=activation_stream,
                swiglu_limit=self.moe_runner_config.swiglu_limit,
                timeline_context=timeline_context,
                timeline_origin=timeline_origin,
                **binding_kwargs,
            )
        else:
            result = launch_bf16_streaming_moe(
                dispatch,
                self.w13_weight,
                self.w2_weight,
                streams=lane_streams,
                drain_stream=drain_stream,
                swiglu_limit=self.moe_runner_config.swiglu_limit,
                timeline_context=timeline_context,
                timeline_origin=timeline_origin,
            )
        # Retain the full result globally only until GPU consumers finish;
        # retain just the ordering fence on the layer itself.
        _deepep_streaming_global_inflight.append(result)
        self._deepep_streaming_inflight[wavefront_slot] = result.epoch_drained
        return result

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            if self.moe_runner_config.is_gated:
                param_data[expert_id][idx] = loaded_weight
            else:
                param_data[expert_id] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
        elif shard_id in ("w1", "w3", "w13"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        assert shard_id in {"w1", "w3", "w13"}

        if is_bias:
            # if this weight is a bias, the last dimension must be the sharded dimension
            shard_dim = -1

        if shard_id in {"w1", "w3"} and self.moe_runner_config.is_gated:
            # non-fused version
            shard_size = expert_data.shape[shard_dim] // 2
        elif shard_id in {"w13"} or (
            shard_id in {"w1", "w3"} and not self.moe_runner_config.is_gated
        ):
            # fused version
            shard_size = expert_data.shape[shard_dim]
        else:
            raise NotImplementedError

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
        switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
        if (
            (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3")
        ) and self.moe_runner_config.is_gated:
            start = shard_size
        else:
            start = 0

        if self.use_padded_loading:
            if _is_cpu and is_bias:
                shard_dim = 1
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                start,
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not self.use_presharded_weights:
                if not is_bias and self.use_triton_kernels:
                    # do not transpose for bias
                    loaded_weight = loaded_weight.transpose(-2, -1)
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

            expert_data = expert_data.narrow(shard_dim, start, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """Load w2 weights for down projection.

        Args:
            expert_data: The expert data tensor to load into
            shard_dim: The dimension to shard along
            shard_id: The shard ID (must be "w2")
            loaded_weight: The weight tensor to load from
            tp_rank: The tensor parallel rank
        """
        if not isinstance(expert_data, torch.Tensor) or not isinstance(
            loaded_weight, torch.Tensor
        ):
            raise ValueError("expert_data and loaded_weight must be torch.Tensor")

        if (
            self.quant_config is not None
            and "modelopt" in self.quant_config.get_name()
            and (expert_data.dim() != 2 or loaded_weight.dim() != 2)
        ):
            raise ValueError(
                f"Expected 2D tensors, got expert_data shape {expert_data.shape} and loaded_weight shape {loaded_weight.shape}"
            )

        if shard_id != "w2":
            raise ValueError(f"shard_id must be 'w2', got {shard_id}")

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        if is_bias:
            # this expert_data is a bias, not weight,
            # for w2_weight_bias in TP, it does not need to be sharded
            shard_size = expert_data.shape[-1]
        else:
            # this parameter is a weight matrix
            # for w2 in TP, it shards the input_features, i.e., shard_dim=2
            shard_size = expert_data.shape[shard_dim]

        if self.use_padded_loading:
            if _is_cpu and is_bias:
                shard_dim = 1
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                0,  # param_data_start
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not is_bias and not self.use_presharded_weights:
                if self.use_triton_kernels:
                    loaded_weight = loaded_weight.transpose(-2, -1)
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _maybe_load_fp8_shared_expert_as_fp4(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        shard_dim: int,
        tp_rank: int,
    ) -> bool:
        if (
            not self._has_fused_shared
            or expert_id < self._num_local_routed
            or self.quant_config is None
            or not getattr(self.quant_config, "is_fp4_experts", False)
            or shard_id not in ("w1", "w2", "w3")
        ):
            return False

        is_weight = (
            "weight" in weight_name
            and "scale" not in weight_name
            and loaded_weight.dtype == torch.float8_e4m3fn
        )
        is_scale = "weight_scale_inv" in weight_name and loaded_weight.dtype in (
            torch.float8_e8m0fnu,
            torch.float32,
        )
        if not is_weight and not is_scale:
            return False

        weight_param = self.w2_weight if shard_id == "w2" else self.w13_weight
        scale_param = (
            self.w2_weight_scale_inv if shard_id == "w2" else self.w13_weight_scale_inv
        )
        if param is not weight_param and param is not scale_param:
            return False

        key = (expert_id, shard_id)
        if is_weight:
            fp8_weight = loaded_weight
            fp8_scale = self._pending_fp8_shared_scales.pop(key, None)
            if fp8_scale is None:
                self._pending_fp8_shared_weights[key] = loaded_weight
                return True
        else:
            fp8_weight = self._pending_fp8_shared_weights.pop(key, None)
            fp8_scale = loaded_weight
            if fp8_weight is None:
                self._pending_fp8_shared_scales[key] = loaded_weight
                return True

        logging.getLogger(__name__).warning_once(
            "Loading FP8 shared expert weights into FP4 fused MoE weights. "
            "The shared expert is quantized at load time and may differ "
            "slightly from a checkpoint that stores shared experts directly "
            "in FP4."
        )

        weight_block_size = getattr(self.quant_config, "weight_block_size", None)
        if weight_block_size is None:
            raise ValueError(
                "Loading FP8 shared expert weights into FP4 fused MoE weights "
                "requires block-FP8 weight_block_size."
            )
        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, weight_block_size
        )

        weight_data = weight_param.data[expert_id]
        scale_data = scale_param.data[expert_id]
        self._load_model_weight_or_group_weight_scale(
            shard_dim=shard_dim,
            expert_data=weight_data,
            shard_id=shard_id,
            loaded_weight=fp4_weight,
            tp_rank=tp_rank,
        )
        self._load_model_weight_or_group_weight_scale(
            shard_dim=shard_dim,
            expert_data=scale_data,
            shard_id=shard_id,
            loaded_weight=fp4_scale,
            tp_rank=tp_rank,
        )
        return True

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        start_idx = self._expert_storage_rank * self._num_local_routed
        end_idx = start_idx + self._num_local_routed
        if start_idx <= expert_id < end_idx:
            return expert_id - start_idx
        elif self._has_fused_shared and expert_id >= self._num_global_routed:
            return expert_id - self._num_global_routed + self._num_local_routed
        else:
            return -1

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: Optional[int],
    ) -> None:
        # if expert_id is None, then
        # all the experts are loaded at the same time
        if (
            not expert_id
            and self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        global_expert_location_metadata = get_global_expert_location_metadata()
        if global_expert_location_metadata is None:
            if not getattr(param, "_sglang_require_global_experts", False):
                expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
                if expert_id == -1:
                    return

            self._weight_loader_impl(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return

        require_global_experts = getattr(param, "_sglang_require_global_experts", False)
        shared_expert_id = (
            expert_id - global_expert_location_metadata.num_logical_experts
            if self._has_fused_shared and expert_id is not None
            else -1
        )
        if 0 <= shared_expert_id < self.num_fused_shared_experts:
            # Checkpoint shared experts start after logical routed experts, while
            # local fused MoE weights store them after physical routed experts.
            if require_global_experts and uses_per_rank_fused_shared_slots():
                physical_expert_ids = [
                    rank * self.num_local_experts
                    + self._num_local_routed
                    + shared_expert_id
                    for rank in range(self.moe_ep_size)
                ]
            else:
                physical_expert_ids = [self._num_global_routed + shared_expert_id]
        else:
            physical_expert_ids = (
                global_expert_location_metadata.logical_to_all_physical(
                    self.layer_id, expert_id, require_global_experts
                )
            )

        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        # WARN: This makes the `expert_id` mean "local" and "global" in different cases
        if not getattr(param, "_sglang_require_global_experts", False):
            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id < 0 or expert_id >= self.num_local_experts:
                return

        if isinstance(
            self.quant_method,
            KTEPWrapperMethod,
        ):
            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
                    return

        self._weight_loader_impl(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )

    def _load_gguf_weight(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        expert_id: int,
        tp_rank: int,
    ) -> bool:
        """Handle GGUF weight loading.

        Args:
            param: The parameter to load the weight into.
            loaded_weight: The weight tensor to load.
            shard_id: The shard ID (w1, w2, or w3).
            expert_id: The expert ID.
            tp_rank: The tensor parallel rank.

        Returns:
            True if the weight was handled as a GGUF weight, False otherwise.
        """
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)

        if is_gguf_weight_type:
            # Store weight type for this expert
            param.weight_type = loaded_weight.item()
            return True

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            if self.moe_tp_size > 1:
                if shard_id in ["w1", "w3", "w2"] and output_dim == 0:
                    shard_size = loaded_weight.size(0) // self.moe_tp_size
                    start_idx = tp_rank * shard_size
                    loaded_weight = loaded_weight.narrow(
                        0, start_idx, shard_size
                    ).clone()

            # Store in data_container with expert/shard info
            if not hasattr(param, "expert_data_map"):
                param.expert_data_map = {}

            key = (expert_id, shard_id)
            param.expert_data_map[key] = loaded_weight
            param.data_container.append(loaded_weight)
            return True

        return False

    def _weight_loader_impl(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        tp_rank = self.moe_tp_rank

        # Special case for GGUF weights
        if self._load_gguf_weight(param, loaded_weight, shard_id, expert_id, tp_rank):
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        if method.__class__.__name__ == "KTEPWrapperMethod":
            method = method.gpu_method

        # For flashinfer TRT-LLM BF16 path, process_weights_after_loading reshapes
        # expert weights into block layout. During weight update, we must restore
        # canonical load-time shapes before copying checkpoint tensors.
        if isinstance(method, UnquantizedFusedMoEMethod):
            method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer=self,
                param=param,
                weight_name=weight_name,
            )
        elif isinstance(method, Fp8MoEMethod) and (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            # Drop the GPU mxfp8 shuffle-index cache on every reload for mxfp8 trtllm, trtllm_routed
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                clear_mxfp8_shuffle_index_cache,
            )

            clear_mxfp8_shuffle_index_cache()

        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MarlinMoE",
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and "zero" not in weight_name
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Flashinfer assumes w31 format for w13_weight. Same for the scales.
        if self.use_flashinfer_trtllm_moe and (
            isinstance(method, ModelOptNvFp4FusedMoEMethod)
            or isinstance(method, Fp8MoEMethod)
            or isinstance(method, UnquantizedFusedMoEMethod)
            or isinstance(method, CompressedTensorsMxInt4MoE)
        ):
            shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if self.use_triton_kernels:
            is_transposed = True
        if is_transposed:
            shard_dim = int(not shard_dim)

        if self._maybe_load_fp8_shared_expert_as_fp4(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            shard_dim=shard_dim,
            tp_rank=tp_rank,
        ):
            return

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust input_scale for e4m3fnuz (AMD)
            if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                loaded_weight = loaded_weight * 2.0

            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                (
                    "compressed" in method.__class__.__name__.lower()
                    or "w4afp8" in self.quant_config.get_name()
                )
                and (param.data[expert_id] != 1).any()
                and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any()
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        if "ModelOpt" in method.__class__.__name__:
            # Determine per-tensor weight scale patterns based on variant
            is_fp4_variant = isinstance(method, ModelOptNvFp4FusedMoEMethod)

            # FP4 uses "weight_scale_2" for per-tensor, FP8 uses "weight_scale" for per-tensor
            per_tensor_conditions = (
                "weight_scale_2" in weight_name
                if is_fp4_variant
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name

            if per_tensor_conditions:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust INT4 column-wise scaling number to e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 0.5

                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust FP8 per-tensor scaling number for e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 2.0

                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        if (
            "bias" in weight_name
            and self.quant_config.quant_description["quant_method"] == "modelslim"
        ):
            self._load_per_channel_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        tp_rank = self.moe_tp_rank

        # Mirror _weight_loader_impl: the trtllm bf16 prep reshapes expert weights
        # into block layout; hot weight updates must restore canonical shapes first.
        method = self.quant_method
        if isinstance(method, KTEPWrapperMethod):
            method = method.gpu_method
        if isinstance(method, UnquantizedFusedMoEMethod):
            method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
                layer=self,
                param=param,
                weight_name=weight_name,
            )

        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            elif "scale" in weight_name:
                param.data.copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO: check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        method = self.quant_method
        if hasattr(self, "scheme"):
            method = self.scheme
        if isinstance(method, Fp8MoEMethod) and (
            get_moe_runner_backend().is_flashinfer_trtllm_routed()
            or get_moe_runner_backend().is_flashinfer_trtllm()
        ):
            # Drop the GPU mxfp8 shuffle-index cache on every reload for mxfp8 trtllm, trtllm_routed
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                clear_mxfp8_shuffle_index_cache,
            )

            clear_mxfp8_shuffle_index_cache()
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MoE",
                    "CompressedTensorsWNA16TritonMoE",
                ]
            )
            and "zero" not in weight_name
            else loaded_weight
        )

        if shard_id not in ("w13", "w2"):
            raise ValueError(f"shard_id must be ['w13','w2'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w13": 1, "w2": 2}
        SHARD_ID_TO_SHARDED_DIM_TRANSPOSE = {"w13": 2, "w2": 1}

        expert_data = param.data
        is_bias = expert_data.dim() == 2

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)

        if self.use_triton_kernels:
            is_transposed = True
        shard_dim = (
            SHARD_ID_TO_SHARDED_DIM[shard_id]
            if not is_transposed
            else SHARD_ID_TO_SHARDED_DIM_TRANSPOSE[shard_id]
        )

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
            return
        else:
            logging.warning(
                f"Unsupported weight_name {weight_name} for FusedMoE weight_loader_fused. Nothing is loaded."
            )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if self._use_ascend_fuseep:
            from sglang.srt.hardware_backend.npu.moe.fuseep import forward_fuseep

            return forward_fuseep(self, hidden_states, topk_output)
        if (
            getattr(self.dispatcher, "streaming_enabled", False)
            and is_in_tc_piecewise_cuda_graph()
        ):
            raise RuntimeError("streaming DeepEP does not support CUDA graph capture")
        if is_in_tc_piecewise_cuda_graph():
            if TopKOutputChecker.format_is_standard(topk_output):
                return moe_forward_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.topk_weights,
                    topk_output.topk_ids,
                    topk_output.router_logits,
                    self.layer_id,
                )
            elif TopKOutputChecker.format_is_bypassed(topk_output):
                return fused_moe_bypassed_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.router_logits,
                    topk_output.topk_config.top_k,
                    topk_output.topk_config.topk_group,
                    topk_output.topk_config.num_expert_group,
                    topk_output.topk_config.correction_bias,
                    topk_output.topk_config.renormalize,
                    self.layer_id,
                    topk_output.topk_config.allow_routed_experts_capture,
                )
            else:
                # Make sure there is torch lib op registration for the whole moe layer
                return self.forward_impl(hidden_states, topk_output)
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

        if getattr(self.dispatcher, "streaming_enabled", False):
            timeline_context = self._deepep_timeline_context("streaming")
            final_hidden_states = self._forward_deepep_streaming(
                hidden_states, topk_output, timeline_context
            )
            return final_hidden_states[..., :origin_hidden_states_dim].contiguous()

        timeline_context = self._deepep_timeline_context("baseline")
        if timeline_context is None:
            dispatch_output = self.dispatcher.dispatch(
                hidden_states=hidden_states, topk_output=topk_output
            )
            combine_input = self.run_moe_core(dispatch_output=dispatch_output)
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                final_hidden_states = self.dispatcher.combine(
                    combine_input=combine_input
                )
                final_hidden_states = final_hidden_states[
                    ..., :origin_hidden_states_dim
                ].contiguous()
        else:
            timeline_origin = torch.cuda.Event(enable_timing=True)
            event_names = (
                ("dispatch_input_ready", "output_ready")
                if timeline_context["profile_detail"] == "arrival"
                else (
                    "dispatch_input_ready",
                    "dispatch_prepare_done",
                    "dispatch_done",
                    "runner_pre_permute_done",
                    "w13_start",
                    "w13_done",
                    "activation_start",
                    "activation_done",
                    "w2_start",
                    "w2_done",
                    "runner_post_permute_done",
                    "runner_fused_done",
                    "gemm_done",
                    "combine_prepare_done",
                    "combine_done",
                    "output_ready",
                )
            )
            timeline = {
                "events": {
                    name: torch.cuda.Event(enable_timing=True) for name in event_names
                },
                "recorded": set(),
                "counters": {},
            }
            timeline_origin.record(torch.cuda.current_stream(hidden_states.device))
            with moe_timeline_scope(timeline):
                dispatch_output = self.dispatcher.dispatch(
                    hidden_states=hidden_states, topk_output=topk_output
                )
                dispatch_buffer_rows = dispatch_output.hidden_states.size(0)
                record_moe_timeline_event("dispatch_done")

                combine_input = self.run_moe_core(
                    dispatch_output=dispatch_output,
                )
                record_moe_timeline_event("gemm_done")

                with use_symmetric_memory(
                    get_tp_group(), disabled=not is_allocation_symmetric()
                ):
                    final_hidden_states = self.dispatcher.combine(
                        combine_input=combine_input
                    )
                    record_moe_timeline_event("combine_done")

                    # TODO: should we add some conditions here?
                    final_hidden_states = final_hidden_states[
                        ..., :origin_hidden_states_dim
                    ].contiguous()
                    record_moe_timeline_event("output_ready")

            if timeline_context["profile_detail"] == "arrival":
                self._emit_arrival_timeline(
                    timeline_context,
                    timeline_origin,
                    timeline["events"]["output_ready"],
                    hidden_states,
                    timeline,
                )
            else:
                self._emit_baseline_timeline(
                    timeline_context,
                    timeline_origin,
                    timeline,
                    hidden_states,
                    topk_output,
                    dispatch_output,
                    dispatch_buffer_rows,
                )

        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    def forward_deferred_finalize(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ):
        assert self.quant_method is not None
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            flashinfer_trtllm_deferred_finalize_context,
        )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        with flashinfer_trtllm_deferred_finalize_context():
            combine_input = self.run_moe_core(dispatch_output=dispatch_output)

        return self.dispatcher.combine(combine_input=combine_input)

    def run_moe_core(
        self, dispatch_output: DispatchOutput | DeepEPStreamingDispatch
    ) -> CombineInput | DeepEPStreamingLayerResult:
        # TODO: consider using symmetric memory
        if isinstance(dispatch_output, DeepEPStreamingDispatch):
            return self._run_deepep_streaming_dispatch(dispatch_output)
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    @classmethod
    def make_expert_params_mapping_fused(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
        ]

    @classmethod
    def make_expert_params_mapping_fused_mxfp4(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
        ckpt_gate_up_proj_scale_name: str,
        ckpt_down_proj_scale_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
            (
                "experts.w13_weight_scale",
                f"experts.{ckpt_gate_up_proj_scale_name}",
                "w13",
            ),
            ("experts.w2_weight_scale", f"experts.{ckpt_down_proj_scale_name}", "w2"),
        ]

    @classmethod
    def make_expert_input_scale_params_mapping(
        cls,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id)
        return [
            (
                "experts.w13_" if shard_id in ["w1", "w3"] else "experts.w2_",
                f"experts.{expert_id}.{shard_id}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id in ["w1", "w2", "w3"]
        ]

    def set_overlap_args(
        self, down_gemm_overlap_args: DownGemmOverlapArgs, meta_overlap_args: dict
    ):
        if hasattr(self, "runner"):
            self.runner.set_overlap_args(down_gemm_overlap_args, meta_overlap_args)
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = down_gemm_overlap_args
            self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        if hasattr(self, "runner"):
            self.runner.clear_overlap_args()
        else:
            # TODO: remove this branch after MoE refactor
            self.down_gemm_overlap_args = None
            self.meta_overlap_args = None

    def materialize_gguf_weights(self) -> None:
        """Process weights after loading, especially for GGUF quantization.

        This materializes GGUF UninitializedParameters from their data_containers.
        """

        for name, param in list(self.named_parameters()):
            is_gguf_weight = getattr(param, "is_gguf_weight", False)

            if is_gguf_weight and isinstance(param, UninitializedParameter):
                data_container = getattr(param, "data_container", [])
                expert_data_map = getattr(param, "expert_data_map", {})
                tensor_shape = getattr(param, "tensor_shape", None)

                if data_container and tensor_shape:
                    # Determine the structure from expert_data_map
                    num_experts = tensor_shape[0]

                    # Collect weights by expert
                    expert_weights = {}
                    for (expert_id, shard_id), weight in expert_data_map.items():
                        if expert_id not in expert_weights:
                            expert_weights[expert_id] = {}
                        expert_weights[expert_id][shard_id] = weight

                    # Build the full tensor
                    if "w13" in name:
                        # w13 is gate+up fused
                        weight_list = []
                        for e in range(num_experts):
                            if e in expert_weights:
                                w1 = expert_weights[e].get("w1")
                                w3 = expert_weights[e].get("w3")

                                if w1 is not None and w3 is not None:
                                    fused = torch.cat([w1, w3], dim=0)
                                    weight_list.append(fused)

                        if weight_list:
                            stacked = torch.stack(weight_list, dim=0)
                            param.materialize(stacked.shape, dtype=stacked.dtype)
                            param.data.copy_(stacked)
                    elif "w2" in name:
                        # w2 is down projection
                        weight_list = []
                        for e in range(num_experts):
                            if e in expert_weights and "w2" in expert_weights[e]:
                                w2_weight = expert_weights[e]["w2"]
                                weight_list.append(w2_weight)

                        if weight_list:
                            stacked = torch.stack(weight_list, dim=0)
                            param.materialize(stacked.shape, dtype=stacked.dtype)
                            param.data.copy_(stacked)


@register_custom_op(out_shape="hidden_states")
def moe_forward_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    # only standard topk output is supported for piecewise cuda graph
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    forward_context = get_tc_piecewise_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]
    return moe_layer.forward_impl(hidden_states, topk_output)


@register_custom_op(out_shape="hidden_states")
def fused_moe_bypassed_piecewise_cuda_graph_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    topk_group: Optional[int],
    num_expert_group: Optional[int],
    correction_bias: Optional[torch.Tensor],
    renormalize: bool,
    layer_id: int,
    allow_routed_experts_capture: bool,
) -> torch.Tensor:
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=TopKConfig(
            top_k=top_k,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            correction_bias=correction_bias,
            renormalize=renormalize,
            allow_routed_experts_capture=allow_routed_experts_capture,
        ),
    )
    forward_context = get_tc_piecewise_forward_context()
    moe_layer = forward_context.moe_layers[layer_id]
    return moe_layer.forward_impl(hidden_states, topk_output)
