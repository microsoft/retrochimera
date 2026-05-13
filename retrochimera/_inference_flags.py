"""Optional inference flags applied via env vars."""
import os
import torch


def apply():
    if os.environ.get("RC_TF32", "0") == "1":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if os.environ.get("RC_CUDNN_BENCHMARK", "0") == "1":
        torch.backends.cudnn.benchmark = True


apply()


def autocast_dtype():
    """Return a torch dtype for autocast based on env, or None to disable autocast.

    RC_AUTOCAST values: '', '0', 'off' -> None; 'bf16'/'bfloat16' -> torch.bfloat16;
    'fp16'/'half'/'float16' -> torch.float16.
    """
    v = os.environ.get("RC_AUTOCAST", "").lower()
    if v in ("", "0", "off", "none"):
        return None
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "half", "float16"):
        return torch.float16
    return None

