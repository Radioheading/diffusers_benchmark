import argparse
import time

import torch
from diffusers.utils import export_to_video

from utils.compile import maybe_compile_pipeline
from utils.device import resolve_device
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
        default=None,
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

    pipe = load_wan_pipeline(
        model_id=args.model_id,
        dtype=dtype,
        vae_dtype=torch.float32,
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
    print(f"[ok] saved video to {args.output} using device={device} dtype={dtype}")


if __name__ == "__main__":
    main()
