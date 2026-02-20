import os
import platform
import subprocess


def _cuda_is_available() -> bool:
    """Uses nvidia-smi to check whether a CUDA-enabled GPU is available."""
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return "CUDA" in result.stdout.decode()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _metal_is_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        p = subprocess.run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return "Metal Support" in p.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _get_env_as_int(name: str, default: int = 0) -> int:
    env = os.environ.get(name, default)
    try:
        env_as_int = int(env)
    except ValueError:
        msg = f"Expected {name} to be an integer, got {env} instead"
        raise ValueError(msg)
    return env_as_int


if _get_env_as_int("MPP_GPU", default=_cuda_is_available()):
    try:
        import cupy as np
    except ImportError:
        import numpy as np
else:
    import numpy as np

numpy = np
