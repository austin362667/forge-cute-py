import pytest

from conftest import require_cuda


def _ops(torch_module):
    if not hasattr(torch_module.ops, "forge_cute_py"):
        pytest.skip("torch.ops.forge_cute_py namespace not registered")
    return torch_module.ops.forge_cute_py


def test_copy_transpose_cpu_raises():
    torch = require_cuda()
    ops = _ops(torch)
    if not hasattr(ops, "copy_transpose"):
        pytest.skip("copy_transpose not registered")
    x_cpu = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    with pytest.raises((NotImplementedError, RuntimeError, AssertionError, ValueError)):
        ops.copy_transpose(x_cpu, 16)


def test_reduce_sum_cpu_raises():
    torch = require_cuda()
    ops = _ops(torch)
    if not hasattr(ops, "reduce_sum"):
        pytest.skip("reduce_sum not registered")
    x_cpu = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    with pytest.raises((NotImplementedError, RuntimeError, AssertionError, ValueError)):
        ops.reduce_sum(x_cpu, -1, "shfl")


def test_softmax_online_cpu_raises():
    torch = require_cuda()
    ops = _ops(torch)
    if not hasattr(ops, "softmax_online"):
        pytest.skip("softmax_online not registered")
    x_cpu = torch.randn(4, 8, device="cpu", dtype=torch.float32)
    with pytest.raises((NotImplementedError, RuntimeError, AssertionError, ValueError)):
        ops.softmax_online(x_cpu, -1)
