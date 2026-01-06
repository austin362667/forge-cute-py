import pytest

from conftest import require_cuda


def _copy_transpose_op(torch):
    if not hasattr(torch.ops, "forge_cute_py") or not hasattr(
        torch.ops.forge_cute_py, "copy_transpose"
    ):
        pytest.skip("torch.ops.forge_cute_py.copy_transpose not registered")
    return torch.ops.forge_cute_py.copy_transpose


@pytest.mark.parametrize("shape", [(4, 8), (16, 32)])
@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_copy_transpose_matches_reference(shape, tile, dtype):
    torch = require_cuda()
    op = _copy_transpose_op(torch)
    x = torch.arange(0, shape[0] * shape[1], device="cuda", dtype=getattr(torch, dtype))
    x = x.reshape(*shape)
    y = op(x, tile)
    y_ref = x.transpose(-2, -1).contiguous()
    torch.testing.assert_close(y, y_ref)
    assert torch.isfinite(y).all()
