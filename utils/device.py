import os

import torch


def normalize_device_name(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "gpu":
        return "cuda"
    if requested.startswith("gpu:"):
        return "cuda:" + requested.split(":", 1)[1]
    return requested


def backend_hint() -> str:
    hints = []
    if getattr(torch.version, "hip", None) is None:
        hints.append(
            "PyTorch is a CUDA build (not ROCm). Install a ROCm-enabled PyTorch wheel in this environment."
        )
    if os.path.exists("/dev/kfd") and not os.access("/dev/kfd", os.R_OK | os.W_OK):
        hints.append(
            "Current user cannot access /dev/kfd. Add the user to the 'render' (and often 'video') group, then relog."
        )
    if not hints:
        hints.append("No GPU backend is visible to this PyTorch runtime.")
    return " ".join(hints)


def resolve_device(requested: str) -> torch.device:
    requested = normalize_device_name(requested)
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    # Explicit GPU request (works for CUDA and ROCm builds)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    raise RuntimeError(
        f"Requested GPU device '{requested}', but torch.cuda.is_available() is False. {backend_hint()}"
    )
