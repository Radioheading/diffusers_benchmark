import argparse
import time

import torch

from utils.compile import maybe_compile_pipeline
from utils.device import resolve_device
from utils.fp8 import (
    FP8_FASTEST_QUANT_TYPE,
    FP8_SAFE_QUANT_TYPE,
    load_qwen_pipeline,
    is_fully_black_image,
)
from utils.generation import run_inference

MODEL_ID = "Qwen/Qwen-Image"
POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图.",
}
DEFAULT_PROMPT = (
    'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," '
    'with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful '
    'Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, gpu, gpu:0, ...")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", default="qwenimage_compiled.png")
    parser.add_argument("--height", type=int, default=1328, help="output image height")
    parser.add_argument("--width", type=int, default=1328, help="output image width")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-lang",
        choices=("en", "zh", "none"),
        default="en",
        help="language key for appending positive_magic suffix; use 'none' to disable suffix",
    )
    parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="negative prompt text; keep a single space to match Qwen reference usage",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="classifier-free guidance scale for QwenImagePipeline",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help=(
            "random seed for diffusion latents. Set this for reproducible output; "
            "use the same value across CPU/GPU runs."
        ),
    )
    parser.add_argument(
        "--generator-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "device for torch.Generator when --seed is set. "
            "For CPU/GPU parity, keep this as 'cpu' (Diffusers samples on CPU then copies to target device)."
        ),
    )
    parser.add_argument(
        "--compile",
        dest="torch_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable torch.compile for supported pipeline modules",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode (used when supported by your PyTorch build); use 'disable' to skip torch.compile",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to compile with fullgraph=True when supported",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "fp8_e4m3"),
        default="none",
        help="quantization mode: fp8_e4m3 uses dynamic activation + weight float8 e4m3 via TorchAO",
    )
    parser.add_argument(
        "--fp8-quant-type",
        default=FP8_FASTEST_QUANT_TYPE,
        help=(
            "TorchAO FP8 quant type to use for --quantization fp8_e4m3. "
            "Fast default is float8dq_e4m3_row; script auto-retries with float8dq_e4m3 if output is fully black"
        ),
    )
    return parser


def run_warmup_and_measured(pipe, args, device: torch.device, stage_label: str = ""):
    suffix = f" {stage_label}" if stage_label else ""
    if args.prompt_lang == "none":
        final_prompt = args.prompt
    else:
        final_prompt = args.prompt + POSITIVE_MAGIC[args.prompt_lang]
    print(f"[info] warmup inference{suffix} (cold start)")
    _ = run_inference(
        pipe=pipe,
        prompt=final_prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        steps=args.steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        generator_device=args.generator_device,
        run_device=device,
    )
    print(f"[info] measured inference{suffix}")
    measured_start = time.perf_counter()
    image = run_inference(
        pipe=pipe,
        prompt=final_prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        steps=args.steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        generator_device=args.generator_device,
        run_device=device,
    )
    measured_end = time.perf_counter()
    elapsed_s = measured_end - measured_start
    print(f"[timing] measured inference{suffix}: {elapsed_s:.6f} s ({elapsed_s * 1000:.3f} ms)")
    return image


def main() -> None:
    args = build_parser().parse_args()

    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"height and width must be > 0, got height={args.height}, width={args.width}")

    device = resolve_device(args.device)
    dtype = torch.bfloat16
    if device.type != "cuda":
        print(
            f"[warn] Running on {device}. For AMD GPU acceleration, use ROCm PyTorch and ensure /dev/kfd access."
        )
    if args.quantization != "none" and device.type != "cuda":
        raise RuntimeError(
            f"Quantization mode '{args.quantization}' requires CUDA device. Resolved device was: {device}."
        )

    pipe, quantization_used = load_qwen_pipeline(
        model_id=args.model_id,
        dtype=dtype,
        quantization=args.quantization,
        fp8_quant_type=args.fp8_quant_type,
    )
    pipe.to(device)

    compile_mode = args.compile_mode.strip().lower()
    compile_enabled = args.torch_compile and compile_mode not in {"disable", "disabled", "none", "off", "false"}
    maybe_compile_pipeline(
        pipe,
        enabled=compile_enabled,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
    )

    if args.seed is not None:
        print(
            f"[info] using seed={args.seed} generator_device={args.generator_device}. "
            "Warmup and measured runs will reuse the same seed."
        )
    try:
        image = run_warmup_and_measured(pipe=pipe, args=args, device=device)
    except Exception as exc:
        retry_without_compile = args.quantization == "fp8_e4m3" and compile_enabled
        if not retry_without_compile:
            raise
        print(
            f"[warn] FP8 run failed with torch.compile enabled ({type(exc).__name__}: {exc}). "
            "Retrying with torch.compile disabled."
        )
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pipe, quantization_used = load_qwen_pipeline(
            model_id=args.model_id,
            dtype=dtype,
            quantization=args.quantization,
            fp8_quant_type=quantization_used,
        )
        pipe.to(device)
        image = run_warmup_and_measured(
            pipe=pipe,
            args=args,
            device=device,
            stage_label="after compile fallback",
        )

    if args.quantization == "fp8_e4m3" and is_fully_black_image(image):
        print(
            f"[warn] generated image is fully black with quant_type={quantization_used}. "
            "This can happen with unstable FP8 kernels."
        )
        if quantization_used == FP8_FASTEST_QUANT_TYPE:
            fallback_quant_type = FP8_SAFE_QUANT_TYPE
            print(
                f"[info] retrying with safer quant_type={fallback_quant_type} "
                "and torch.compile disabled for reliability."
            )
            del pipe
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pipe, quantization_used = load_qwen_pipeline(
                model_id=args.model_id,
                dtype=dtype,
                quantization=args.quantization,
                fp8_quant_type=fallback_quant_type,
            )
            pipe.to(device)
            image = run_warmup_and_measured(
                pipe=pipe,
                args=args,
                device=device,
                stage_label="after FP8 fallback",
            )
            if is_fully_black_image(image):
                raise RuntimeError(
                    "FP8 retry also produced a fully black image. "
                    "Try --no-compile, fewer steps/resolution, or --quantization none."
                )
        else:
            raise RuntimeError(
                "FP8 run produced a fully black image. "
                f"Try --fp8-quant-type {FP8_SAFE_QUANT_TYPE} or --quantization none."
            )

    image.save(args.output)
    print(f"[ok] saved image to {args.output} using device={device} dtype={dtype} quantization={quantization_used}")


if __name__ == "__main__":
    main()
