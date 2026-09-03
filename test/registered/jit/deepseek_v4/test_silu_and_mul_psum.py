import pytest
import torch
from transformers import AutoConfig

# This checkout predates transformers' built-in Qwen3-ASR registration.  Make
# SGLang's compatibility registration idempotent so the kernel test can import
# against a newer developer environment.
_register_config = AutoConfig.register
AutoConfig.register = lambda model_type, config, exist_ok=False: _register_config(
    model_type, config, exist_ok=True
)
try:
    from sglang.jit_kernel.dsv4.moe import (
        silu_and_mul_contig_post_quant,
        silu_and_mul_psum_post_quant,
    )
finally:
    AutoConfig.register = _register_config


def _valid_rows(psum: list[int], alignment: int) -> torch.Tensor:
    rows: list[int] = []
    previous_end = 0
    for expert, expert_end in enumerate(psum):
        expert_start = (
            0 if expert == 0 else ((previous_end + alignment - 1) // alignment) * alignment
        )
        rows.extend(range(expert_start, expert_end))
        previous_end = expert_end
    return torch.tensor(rows, dtype=torch.long, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("scale_ue8m0", "swizzle", "swiglu_limit"),
    (
        (False, False, None),
        (True, False, None),
        (True, True, None),
        (True, True, 7.0),
    ),
)
def test_psum_activation_matches_contiguous_on_valid_rows(
    scale_ue8m0: bool, swizzle: bool, swiglu_limit: float | None
):
    torch.manual_seed(7)
    alignment = 128
    psum_values = [3, 128, 133, 258]
    num_rows = 384
    hidden = 512 if scale_ue8m0 else 256
    groups = hidden // 128
    gate_up = torch.randn(
        (num_rows, hidden * 2), device="cuda", dtype=torch.bfloat16
    )
    psum = torch.tensor(psum_values, device="cuda", dtype=torch.int32)

    expected = torch.empty(
        (num_rows, hidden), device="cuda", dtype=torch.float8_e4m3fn
    )
    actual = torch.zeros_like(expected)
    if scale_ue8m0:
        expected_scale_storage = torch.empty(
            (groups // 4, num_rows), device="cuda", dtype=torch.int32
        )
        actual_scale_storage = torch.zeros_like(expected_scale_storage)
        expected_scale = expected_scale_storage.transpose(0, 1)
        actual_scale = actual_scale_storage.transpose(0, 1)
    else:
        expected_scale = torch.empty(
            (num_rows, groups), device="cuda", dtype=torch.float32
        )
        actual_scale = torch.zeros_like(expected_scale)

    silu_and_mul_contig_post_quant(
        gate_up,
        expected,
        expected_scale,
        128,
        scale_ue8m0=scale_ue8m0,
        transposed=scale_ue8m0,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
    )
    silu_and_mul_psum_post_quant(
        gate_up,
        actual,
        actual_scale,
        128,
        psum,
        expert_alignment=alignment,
        scale_ue8m0=scale_ue8m0,
        transposed=scale_ue8m0,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        use_pdl=False,
    )
    torch.cuda.synchronize()

    valid = _valid_rows(psum_values, alignment)
    assert torch.equal(actual[valid], expected[valid])
    assert torch.equal(actual_scale[valid], expected_scale[valid])

    invalid = torch.ones(num_rows, dtype=torch.bool, device="cuda")
    invalid[valid] = False
    assert torch.count_nonzero(actual[invalid]) == 0
    assert torch.count_nonzero(actual_scale[invalid]) == 0
