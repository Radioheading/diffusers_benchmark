import importlib.util
from typing import Any

import torch

FP8_SAFE_QUANT_TYPE = "float8dq_e4m3"
FP8_FASTEST_QUANT_TYPE = "float8dq_e4m3_row"
FP8_FALLBACK_QUANT_TYPES = ("float8dq",)


def preferred_e4m3_dtype() -> torch.dtype:
    # ROCm kernels use FNUZ encoding for float8 e4m3.
    if getattr(torch.version, "hip", None) is not None and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def build_torchao_config(quant_type: str):
    from diffusers import TorchAoConfig

    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        from torchao.quantization.observer import PerRow, PerTensor
    except Exception:
        return TorchAoConfig(quant_type)

    fp8_dtype = preferred_e4m3_dtype()

    if quant_type == "float8dq_e4m3_row":
        return TorchAoConfig(
            Float8DynamicActivationFloat8WeightConfig(
                activation_dtype=fp8_dtype,
                weight_dtype=fp8_dtype,
                granularity=PerRow(),
            )
        )
    if quant_type == "float8dq_e4m3":
        return TorchAoConfig(
            Float8DynamicActivationFloat8WeightConfig(
                activation_dtype=fp8_dtype,
                weight_dtype=fp8_dtype,
                granularity=PerTensor(),
            )
        )
    if quant_type == "float8dq":
        return TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())

    return TorchAoConfig(quant_type)


def load_qwen_pipeline(
    model_id: str,
    dtype: torch.dtype,
    quantization: str,
    fp8_quant_type: str,
) -> tuple[Any, str]:
    from diffusers import QwenImagePipeline

    if quantization == "none":
        pipe = QwenImagePipeline.from_pretrained(model_id, torch_dtype=dtype)
        return pipe, "none"
    if quantization != "fp8_e4m3":
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    if importlib.util.find_spec("torchao") is None:
        raise RuntimeError(
            "FP8 quantization requires torchao. Install it first, for example: pip install torchao"
        )

    from diffusers import QwenImageTransformer2DModel

    candidate_quant_types = [fp8_quant_type]
    if fp8_quant_type == FP8_FASTEST_QUANT_TYPE:
        candidate_quant_types.extend((FP8_SAFE_QUANT_TYPE, *FP8_FALLBACK_QUANT_TYPES))

    attempted = []
    last_error = None
    for quant_type in candidate_quant_types:
        attempted.append(quant_type)
        try:
            quant_config = build_torchao_config(quant_type)
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
            )
            pipe = QwenImagePipeline.from_pretrained(
                model_id,
                transformer=transformer,
                torch_dtype=dtype,
            )
            return pipe, quant_type
        except Exception as exc:
            print(f"[warn] failed to load fp8 quantization with quant_type={quant_type}: {exc}")
            last_error = exc

    raise RuntimeError(
        f"Unable to load FP8 quantization for {model_id}. Tried quantization types: {attempted}. "
        f"Last error: {last_error}"
    )


def is_fully_black_image(image) -> bool:
    try:
        extrema = image.convert("RGB").getextrema()
    except Exception:
        return False
    return all(channel_min == 0 and channel_max == 0 for channel_min, channel_max in extrema)
