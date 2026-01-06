import pytest
import torch

from conftest import require_cuda


def _reduce_sum_op(torch):
    if not hasattr(torch.ops, "forge_cute_py") or not hasattr(
        torch.ops.forge_cute_py, "reduce_sum"
    ):
        pytest.skip("torch.ops.forge_cute_py.reduce_sum not registered")
    return torch.ops.forge_cute_py.reduce_sum


def _tolerance(dtype):
    if dtype in (torch.bfloat16, torch.float16):
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 1e-4, "rtol": 1e-4}


@pytest.mark.parametrize("shape,dim", [((4, 8), -1), ((8, 4), 0)])
@pytest.mark.parametrize("dtype", [pytest.param("float16"), pytest.param("float32")])
@pytest.mark.parametrize("variant", ["naive", "shfl"])
def test_reduce_sum_matches_reference(shape, dim, dtype, variant):
    torch = require_cuda()
    op = _reduce_sum_op(torch)
    x = torch.randn(*shape, device="cuda", dtype=getattr(torch, dtype))
    try:
        y = op(x, dim, variant)
    except NotImplementedError:
        pytest.skip(f"reduce_sum variant {variant} not implemented")
    if x.dtype in (torch.bfloat16, torch.float16):
        y_ref = x.float().sum(dim=dim).to(x.dtype)
    else:
        y_ref = x.sum(dim=dim)
    torch.testing.assert_close(y, y_ref, **_tolerance(x.dtype))
    assert torch.isfinite(y).all()


def test_reduce_sum_torch_compile():
    torch = require_cuda()
    op = _reduce_sum_op(torch)
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    unsupported_exc = ()
    try:
        from torch._dynamo.exc import Unsupported as DynamoUnsupported

        unsupported_exc = (DynamoUnsupported,)
    except Exception:
        unsupported_exc = ()
    try:
        compiled = torch.compile(lambda x: op(x, -1, "shfl"), fullgraph=True)
    except Exception as exc:
        pytest.skip(f"torch.compile not available for reduce_sum: {exc}")
    x = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    try:
        y = compiled(x)
    except unsupported_exc as exc:
        pytest.skip(f"torch.compile unsupported for reduce_sum op: {exc}")
    except NotImplementedError:
        pytest.skip("reduce_sum shfl variant not implemented")
    y_ref = x.float().sum(dim=-1).to(x.dtype)
    torch.testing.assert_close(y, y_ref, atol=1e-2, rtol=1e-2)
