import importlib.util
from typing import Any

import torch
from diffusers import AutoencoderKLWan, WanPipeline

from utils.fp8 import FP8_FALLBACK_QUANT_TYPES, FP8_FASTEST_QUANT_TYPE, FP8_SAFE_QUANT_TYPE, build_torchao_config

LIGHTX2V_REPO = "Kijai/WanVideo_comfy"
LIGHTX2V_HIGH_WEIGHT_CANDIDATES = (
    "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
    "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors",
    "Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors",
)
LIGHTX2V_LOW_WEIGHT_CANDIDATES = (
    "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
    "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors",
    "Wan22-Lightning/Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors",
)
LIGHTX2V_ADAPTER_1 = "lightning"
LIGHTX2V_ADAPTER_2 = "lightning_2"


def load_wan_pipeline(
    model_id: str,
    dtype: torch.dtype,
    vae_dtype: torch.dtype = torch.float32,
    quantization: str = "none",
    fp8_quant_type: str = FP8_FASTEST_QUANT_TYPE,
) -> tuple[Any, str]:
    if quantization == "none":
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
        )
        return (
            WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=dtype,
            ),
            "none",
        )

    if quantization != "fp8_e4m3":
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    if importlib.util.find_spec("torchao") is None:
        raise RuntimeError(
            "FP8 quantization requires torchao. Install it first, for example: pip install torchao"
        )

    from diffusers import WanTransformer3DModel

    candidate_quant_types = [fp8_quant_type]
    if fp8_quant_type == FP8_FASTEST_QUANT_TYPE:
        candidate_quant_types.extend((FP8_SAFE_QUANT_TYPE, *FP8_FALLBACK_QUANT_TYPES))

    attempted = []
    last_error = None
    for quant_type in candidate_quant_types:
        attempted.append(quant_type)
        try:
            quant_config = build_torchao_config(quant_type)
            transformer = WanTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
            )
            transformer_2 = WanTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer_2",
                quantization_config=quant_config,
                torch_dtype=dtype,
            )
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=vae_dtype,
            )
            pipe = WanPipeline.from_pretrained(
                model_id,
                vae=vae,
                transformer=transformer,
                transformer_2=transformer_2,
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


def load_lora_with_fallback(
    pipe: Any,
    repo_id: str,
    weight_name_candidates: tuple[str, ...],
    adapter_name: str,
    **kwargs,
) -> str:
    last_error = None
    for weight_name in weight_name_candidates:
        try:
            pipe.load_lora_weights(
                repo_id,
                weight_name=weight_name,
                adapter_name=adapter_name,
                **kwargs,
            )
            return weight_name
        except Exception as exc:
            last_error = exc
            print(f"[warn] failed to load LoRA '{weight_name}' for adapter '{adapter_name}': {exc}")

    raise RuntimeError(
        f"Unable to load LoRA for adapter '{adapter_name}' from repo '{repo_id}'. "
        f"Tried: {list(weight_name_candidates)}. Last error: {last_error}"
    )


def enable_lightx2v(pipe: Any) -> None:
    high_weight = load_lora_with_fallback(
        pipe,
        repo_id=LIGHTX2V_REPO,
        weight_name_candidates=LIGHTX2V_HIGH_WEIGHT_CANDIDATES,
        adapter_name=LIGHTX2V_ADAPTER_1,
    )
    low_weight = load_lora_with_fallback(
        pipe,
        repo_id=LIGHTX2V_REPO,
        weight_name_candidates=LIGHTX2V_LOW_WEIGHT_CANDIDATES,
        adapter_name=LIGHTX2V_ADAPTER_2,
        load_into_transformer_2=True,
    )
    pipe.set_adapters(
        [LIGHTX2V_ADAPTER_1, LIGHTX2V_ADAPTER_2],
        adapter_weights=[1.0, 1.0],
    )
    print(
        f"[info] loaded --lightx2v LoRAs: "
        f"{LIGHTX2V_ADAPTER_1}={high_weight}, {LIGHTX2V_ADAPTER_2}={low_weight}"
    )


def make_generator(
    seed: int | None,
    generator_device: str,
    run_device: torch.device,
) -> torch.Generator | None:
    if seed is None:
        return None
    if generator_device == "cuda" and run_device.type != "cuda":
        raise ValueError(
            f"--generator-device cuda requires a CUDA run device, but resolved device was: {run_device}"
        )
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def run_wan_inference(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    frames: int,
    steps: int,
    guidance_scale: float,
    guidance_scale_2: float,
    seed: int | None,
    generator_device: str,
    run_device: torch.device,
):
    generator = make_generator(seed=seed, generator_device=generator_device, run_device=run_device)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=frames,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        num_inference_steps=steps,
        generator=generator,
    ).frames[0]
