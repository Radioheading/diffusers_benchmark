import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from diffusers.utils import export_to_video

from utils.compile import maybe_compile_pipeline
from utils.device import resolve_device
from utils.fp8 import FP8_FASTEST_QUANT_TYPE
from utils.wan import enable_lightx2v, load_wan_pipeline, run_wan_inference

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_STEPS = 40
LIGHTX2V_STEPS = 4
LIGHTX2V_GUIDANCE_SCALE = 1.0
LIGHTX2V_GUIDANCE_SCALE_2 = 1.0
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

WAN_VAE_SCALE_FACTOR_SPATIAL = 8
WAN_VAE_SCALE_FACTOR_TEMPORAL = 4
WAN_PATCH_SIZE = (1, 2, 2)
WAN_IN_CHANNELS = 16
WAN_TEXT_DIM = 4096
WAN_INNER_DIM = 40 * 128
WAN_FFN_DIM = 13824
WAN_NUM_LAYERS = 40
WAN_DEFAULT_TEXT_SEQ_LEN = 512


def derive_wan_linear_gemm_shapes(
    height: int = 720,
    width: int = 1280,
    frames: int = 81,
    text_seq_len: int = WAN_DEFAULT_TEXT_SEQ_LEN,
) -> dict:
    if height <= 0 or width <= 0 or frames <= 0:
        raise ValueError(f"height/width/frames must be > 0, got {height}/{width}/{frames}")
    if text_seq_len <= 0:
        raise ValueError(f"text_seq_len must be > 0, got {text_seq_len}")

    # Wan pipeline rounds num_frames to satisfy (num_frames - 1) % 4 == 0.
    effective_frames = frames
    if effective_frames % WAN_VAE_SCALE_FACTOR_TEMPORAL != 1:
        effective_frames = effective_frames // WAN_VAE_SCALE_FACTOR_TEMPORAL * WAN_VAE_SCALE_FACTOR_TEMPORAL + 1
    effective_frames = max(effective_frames, 1)

    latent_frames = (effective_frames - 1) // WAN_VAE_SCALE_FACTOR_TEMPORAL + 1
    latent_height = int(height) // WAN_VAE_SCALE_FACTOR_SPATIAL
    latent_width = int(width) // WAN_VAE_SCALE_FACTOR_SPATIAL

    patch_t, patch_h, patch_w = WAN_PATCH_SIZE
    post_patch_frames = latent_frames // patch_t
    post_patch_height = latent_height // patch_h
    post_patch_width = latent_width // patch_w
    latent_seq_len = post_patch_frames * post_patch_height * post_patch_width

    shapes = [
        {
            "name": "wan.text_proj_in",
            "m": text_seq_len,
            "k": WAN_TEXT_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": 1,
        },
        {
            "name": "wan.text_proj_out",
            "m": text_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": 1,
        },
        {
            "name": "wan.self_attn_qkv",
            "m": latent_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": WAN_NUM_LAYERS * 3,
        },
        {
            "name": "wan.self_attn_out",
            "m": latent_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": WAN_NUM_LAYERS,
        },
        {
            "name": "wan.cross_attn_q",
            "m": latent_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": WAN_NUM_LAYERS,
        },
        {
            "name": "wan.cross_attn_kv",
            "m": text_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": WAN_NUM_LAYERS * 2,
        },
        {
            "name": "wan.cross_attn_out",
            "m": latent_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": WAN_NUM_LAYERS,
        },
        {
            "name": "wan.ffn_in",
            "m": latent_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_FFN_DIM,
            "occurrences": WAN_NUM_LAYERS,
        },
        {
            "name": "wan.ffn_out",
            "m": latent_seq_len,
            "k": WAN_FFN_DIM,
            "n": WAN_INNER_DIM,
            "occurrences": WAN_NUM_LAYERS,
        },
        {
            "name": "wan.proj_out",
            "m": latent_seq_len,
            "k": WAN_INNER_DIM,
            "n": WAN_IN_CHANNELS * patch_t * patch_h * patch_w,
            "occurrences": 1,
        },
    ]

    return {
        "model": MODEL_ID,
        "height": height,
        "width": width,
        "frames": frames,
        "effective_frames": effective_frames,
        "latent_frames": latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "latent_seq_len": latent_seq_len,
        "text_seq_len": text_seq_len,
        "shapes": shapes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, gpu, gpu:0, ...")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", "--num-frames", dest="frames", type=int, default=81)
    parser.add_argument(
        "--steps",
        "--num-inference-steps",
        dest="steps",
        type=int,
        default=None,
        help=(
            f"number of denoising steps; default is {DEFAULT_STEPS}, "
            f"or {LIGHTX2V_STEPS} when --lightx2v is enabled"
        ),
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--output", default="t2v_out.mp4")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for diffusion latents; leave unset to use random sampling",
    )
    parser.add_argument(
        "--generator-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "device for torch.Generator when --seed is set. "
            "For CPU/GPU parity, keep this as 'cpu'."
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
        "--lightx2v",
        action="store_true",
        help=(
            "enable Wan2.2 Lightning x2v LoRA adapters from Kijai/WanVideo_comfy; "
            f"default steps become {LIGHTX2V_STEPS} and guidance scales are fixed to "
            f"{LIGHTX2V_GUIDANCE_SCALE}/{LIGHTX2V_GUIDANCE_SCALE_2}"
        ),
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
            "Fast default is float8dq_e4m3_row; fallback tries safer variants on load failure"
        ),
    )
    parser.add_argument(
        "--quant-t2-only",
        action="store_true",
        help=(
            "with --quantization fp8_e4m3, quantize only transformer_2 to FP8 "
            "(keep transformer in torch_dtype)"
        ),
    )
    return parser


def run_warmup_and_measured(pipe, args, device: torch.device, stage_label: str = ""):
    suffix = f" {stage_label}" if stage_label else ""
    print(f"[info] warmup inference{suffix} (cold start)")
    _ = run_wan_inference(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        frames=args.frames,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        seed=args.seed,
        generator_device=args.generator_device,
        run_device=device,
    )
    print(f"[info] measured inference{suffix}")
    measured_start = time.perf_counter()
    output_frames = run_wan_inference(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        frames=args.frames,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        seed=args.seed,
        generator_device=args.generator_device,
        run_device=device,
    )
    measured_end = time.perf_counter()
    elapsed_s = measured_end - measured_start
    print(f"[timing] measured inference{suffix}: {elapsed_s:.6f} s ({elapsed_s * 1000:.3f} ms)")
    return output_frames


def main() -> None:
    args = build_parser().parse_args()
    if args.steps is None:
        args.steps = LIGHTX2V_STEPS if args.lightx2v else DEFAULT_STEPS
    if args.lightx2v:
        if args.guidance_scale != LIGHTX2V_GUIDANCE_SCALE or args.guidance_scale_2 != LIGHTX2V_GUIDANCE_SCALE_2:
            print(
                f"[info] --lightx2v overrides guidance scales to "
                f"{LIGHTX2V_GUIDANCE_SCALE}/{LIGHTX2V_GUIDANCE_SCALE_2}."
            )
        args.guidance_scale = LIGHTX2V_GUIDANCE_SCALE
        args.guidance_scale_2 = LIGHTX2V_GUIDANCE_SCALE_2

    if args.height <= 0 or args.width <= 0:
        raise ValueError(f"height and width must be > 0, got height={args.height}, width={args.width}")
    if args.frames <= 0:
        raise ValueError(f"frames must be > 0, got frames={args.frames}")
    if args.steps <= 0:
        raise ValueError(f"steps must be > 0, got steps={args.steps}")
    if args.fps <= 0:
        raise ValueError(f"fps must be > 0, got fps={args.fps}")

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
    if args.quant_t2_only and args.quantization != "fp8_e4m3":
        raise ValueError("--quant-t2-only requires --quantization fp8_e4m3")

    pipe, quantization_used = load_wan_pipeline(
        model_id=args.model_id,
        dtype=dtype,
        vae_dtype=torch.float32,
        quantization=args.quantization,
        fp8_quant_type=args.fp8_quant_type,
        quant_t2_only=args.quant_t2_only,
    )
    pipe.to(device)
    if args.lightx2v:
        print("[info] enabling --lightx2v LoRA adapters")
        enable_lightx2v(pipe)

    compile_mode = args.compile_mode.strip().lower()
    compile_enabled = args.torch_compile and compile_mode not in {"disable", "disabled", "none", "off", "false"}
    maybe_compile_pipeline(
        pipe,
        enabled=compile_enabled,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
        module_names=("transformer",),
    )

    if args.seed is not None:
        print(
            f"[info] using seed={args.seed} generator_device={args.generator_device}. "
            "Warmup and measured runs will reuse the same seed."
        )

    output_frames = run_warmup_and_measured(pipe=pipe, args=args, device=device)
    export_to_video(output_frames, args.output, fps=args.fps)
    print(f"[ok] saved video to {args.output} using device={device} dtype={dtype} quantization={quantization_used}")


if __name__ == "__main__":
    main()
