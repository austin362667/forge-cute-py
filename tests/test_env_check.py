import importlib.util
import subprocess
import sys

import pytest

from conftest import require_cuda


def test_env_check_module_runs():
    require_cuda()
    if importlib.util.find_spec("forge_cute_py.env_check") is None:
        pytest.skip("forge_cute_py.env_check not implemented yet")
    result = subprocess.run(
        [sys.executable, "-m", "forge_cute_py.env_check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            "env_check failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
