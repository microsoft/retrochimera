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
