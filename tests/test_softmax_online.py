import pytest
import torch

from conftest import require_cuda


def _softmax_op(torch_module):
    if not hasattr(torch_module.ops, "forge_cute_py") or not hasattr(
        torch_module.ops.forge_cute_py, "softmax_online"
    ):
        pytest.skip("torch.ops.forge_cute_py.softmax_online not registered")
    return torch_module.ops.forge_cute_py.softmax_online


def _tolerance(dtype):
    if dtype == torch.bfloat16:
        return {"atol": 1e-2, "rtol": 1e-2}
    if dtype == torch.float16:
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 1e-4, "rtol": 1e-4}


@pytest.mark.parametrize("shape", [(4, 8), (2, 128)])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_softmax_online_matches_reference(shape, input_dtype):
    torch_mod = require_cuda()
    op = _softmax_op(torch_mod)
    x = (0.1 * torch_mod.randn(*shape, device="cuda", dtype=input_dtype)).requires_grad_(True)
    y = op(x, -1)
    y_ref = torch_mod.softmax(x.float(), dim=-1).to(x.dtype)
    torch_mod.testing.assert_close(y, y_ref, **_tolerance(x.dtype))
    assert torch_mod.isfinite(y).all()
    assert y.shape == x.shape
    assert y.dtype == x.dtype


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_online_properties(input_dtype):
    torch_mod = require_cuda()
    op = _softmax_op(torch_mod)
    x = torch_mod.randn(16, 256, device="cuda", dtype=input_dtype)
    y = op(x, -1)
    sums = torch_mod.sum(y, dim=-1)
    torch_mod.testing.assert_close(sums, torch_mod.ones_like(sums), atol=1e-3, rtol=1e-3)
    assert (y >= 0).all()
    assert (y <= 1).all()


def test_softmax_online_translation_invariance():
    torch_mod = require_cuda()
    op = _softmax_op(torch_mod)
    x = torch_mod.randn(8, 128, device="cuda", dtype=torch.float32)
    y = op(x, -1)
    y_shifted = op(x + 100.0, -1)
    torch_mod.testing.assert_close(y, y_shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_softmax_online_extreme_values(input_dtype):
    torch_mod = require_cuda()
    op = _softmax_op(torch_mod)
    m, n = 8, 256
    x_large = torch_mod.full((m, n), 10.0, device="cuda", dtype=input_dtype)
    out_large = op(x_large, -1)
    expected = torch_mod.full_like(out_large, 1.0 / n)
    torch_mod.testing.assert_close(out_large, expected, atol=1e-3, rtol=1e-3)
    x_small = torch_mod.full((m, n), -10.0, device="cuda", dtype=input_dtype)
    out_small = op(x_small, -1)
    torch_mod.testing.assert_close(out_small, expected, atol=1e-3, rtol=1e-3)
    x_mixed = torch_mod.zeros((m, n), device="cuda", dtype=input_dtype)
    x_mixed[:, 0] = 10.0
    x_mixed[:, 1:] = -10.0
    out_mixed = op(x_mixed, -1)
    assert (out_mixed[:, 0] > 0.99).all()
    assert (out_mixed[:, 1:] < 0.01).all()


def test_softmax_online_torch_compile():
    torch_mod = require_cuda()
    op = _softmax_op(torch_mod)
    if not hasattr(torch_mod, "compile"):
        pytest.skip("torch.compile not available")
    unsupported_exc = ()
    try:
        from torch._dynamo.exc import Unsupported as DynamoUnsupported

        unsupported_exc = (DynamoUnsupported,)
    except Exception:
        unsupported_exc = ()
    try:
        compiled = torch_mod.compile(lambda x: op(x, -1), fullgraph=True)
    except Exception as exc:
        pytest.skip(f"torch.compile not available for softmax_online: {exc}")
    x = torch_mod.randn(4, 16, device="cuda", dtype=torch.float16)
    try:
        y = compiled(x)
    except unsupported_exc as exc:
        pytest.skip(f"torch.compile unsupported for softmax_online op: {exc}")
    y_ref = torch_mod.softmax(x.float(), dim=-1).to(x.dtype)
    torch_mod.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)
