import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn


def _module_is_under_path(mod, root: Path) -> bool:
    module_file = getattr(mod, "__file__", None)
    if module_file is not None:
        resolved = Path(module_file).resolve()
        if resolved == root or root in resolved.parents:
            return True

    module_paths = getattr(mod, "__path__", None)
    if module_paths:
        for module_path in module_paths:
            resolved = Path(module_path).resolve()
            if resolved == root or root in resolved.parents:
                return True

    return False


# Some environments preload similarly named top-level packages.
# Ensure we import this repo's "diffusers_tests" package.
local_tests_root = REPO_ROOT / "diffusers_tests"
loaded_tests = sys.modules.get("diffusers_tests")
if loaded_tests is not None and not _module_is_under_path(loaded_tests, local_tests_root):
    for module_name in tuple(sys.modules):
        if module_name == "diffusers_tests" or module_name.startswith("diffusers_tests."):
            del sys.modules[module_name]

from diffusers_tests.gen.benchmark_qwen import DEFAULT_PROMPT as QWEN_DEFAULT_PROMPT
from diffusers_tests.gen.benchmark_qwen import derive_qwen_linear_gemm_shapes
from diffusers_tests.gen.benchmark_wan import derive_wan_linear_gemm_shapes

MODE_BF16 = "bf16"
MODE_BF16_COMPILE = "bf16_compile"
MODE_FP8_COMPILE = "fp8_compile"
ALL_MODES = (MODE_BF16, MODE_BF16_COMPILE, MODE_FP8_COMPILE)

BENCH_TARGET_GEMM = "gemm"
BENCH_TARGET_LINEAR = "linear"
BENCH_TARGET_CHOICES = (BENCH_TARGET_GEMM, BENCH_TARGET_LINEAR)

FP8_SCALING_AUTO = "auto"
FP8_SCALING_PER_TENSOR = "per_tensor"
FP8_SCALING_PER_ROW = "per_row"
FP8_SCALING_CHOICES = (FP8_SCALING_AUTO, FP8_SCALING_PER_TENSOR, FP8_SCALING_PER_ROW)


@dataclass
class GemmShape:
    family: str
    name: str
    m: int
    k: int
    n: int
    occurrences: int


@dataclass
class BenchResult:
    family: str
    name: str
    m: int
    k: int
    n: int
    bench_target: str
    mode: str
    status: str
    median_ms: float | None
    mean_ms: float | None
    p90_ms: float | None
    tflops: float | None
    note: str


@dataclass(frozen=True)
class FP8Config:
    backend: str
    dtype: torch.dtype
    scaling: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark representative Linear-module shapes from Qwen-Image (1328x1328) and "
            "Wan2.2 (720x1280, 81 frames) with CUDA-event timing."
        )
    )
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda or cuda:0")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=ALL_MODES,
        default=list(ALL_MODES),
        help="benchmark modes to run",
    )
    parser.add_argument(
        "--bench-target",
        choices=BENCH_TARGET_CHOICES,
        default=BENCH_TARGET_LINEAR,
        help="benchmark pure GEMM kernel path or full Linear module forward",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode for compile-based modes",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use fullgraph=True for torch.compile",
    )
    parser.add_argument(
        "--linear-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="benchmark Linear module with/without bias term",
    )
    parser.add_argument(
        "--fp8-scaling",
        choices=FP8_SCALING_CHOICES,
        default=FP8_SCALING_AUTO,
        help=(
            "FP8 scaling strategy for fp8_compile: "
            "'auto' chooses AMD=per_tensor, NVIDIA=per_row."
        ),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("qwen", "wan"),
        default=["qwen", "wan"],
        help="shape families to benchmark",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dedupe shapes by (m, k, n) while summing occurrences",
    )
    parser.add_argument(
        "--max-shapes",
        type=int,
        default=None,
        help="optional cap on number of shapes after dedupe",
    )
    parser.add_argument(
        "--print-shapes-only",
        action="store_true",
        help="print discovered shapes and exit without running kernels",
    )
    parser.add_argument("--csv", type=str, default=None, help="optional csv output path")

    parser.add_argument("--qwen-height", type=int, default=1328)
    parser.add_argument("--qwen-width", type=int, default=1328)
    parser.add_argument("--qwen-prompt", default=QWEN_DEFAULT_PROMPT)
    parser.add_argument("--qwen-prompt-lang", choices=("en", "zh", "none"), default="en")
    parser.add_argument("--qwen-negative-prompt", default=" ")
    parser.add_argument("--qwen-max-sequence-length", type=int, default=512)
    parser.add_argument("--qwen-prompt-seq-len", type=int, default=None)
    parser.add_argument("--qwen-negative-seq-len", type=int, default=None)
    parser.add_argument(
        "--include-qwen-negative",
        action="store_true",
        help="include negative-prompt text-side GEMM shapes",
    )

    parser.add_argument("--wan-height", type=int, default=720)
    parser.add_argument("--wan-width", type=int, default=1280)
    parser.add_argument("--wan-frames", type=int, default=81)
    parser.add_argument("--wan-text-seq-len", type=int, default=512)

    return parser


def _compile_fn(fn, mode: str, fullgraph: bool):
    try:
        return torch.compile(fn, mode=mode, fullgraph=fullgraph)
    except TypeError:
        try:
            return torch.compile(fn, mode=mode)
        except TypeError:
            return torch.compile(fn)


def detect_gpu_backend() -> str:
    if getattr(torch.version, "hip", None) is not None:
        return "amd"
    if getattr(torch.version, "cuda", None) is not None:
        return "nvidia"
    return "unknown"


def resolve_fp8_config(fp8_scaling: str) -> FP8Config:
    backend = detect_gpu_backend()

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("FP8 requires torch.float8_e4m3fn support in this PyTorch build.")

    if backend == "amd" and hasattr(torch, "float8_e4m3fnuz"):
        dtype = torch.float8_e4m3fnuz
        default_scaling = FP8_SCALING_PER_TENSOR
    else:
        dtype = torch.float8_e4m3fn
        default_scaling = FP8_SCALING_PER_ROW

    scaling = default_scaling if fp8_scaling == FP8_SCALING_AUTO else fp8_scaling
    return FP8Config(backend=backend, dtype=dtype, scaling=scaling)


def _tensorwise_scale(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    fp8_max = float(torch.finfo(fp8_dtype).max)
    amax = x.abs().amax().to(dtype=torch.float32)
    scale = amax / fp8_max
    return torch.clamp(scale, min=1e-8)


def _rowwise_scale(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    fp8_max = float(torch.finfo(fp8_dtype).max)
    amax = x.abs().amax(dim=1, keepdim=True).to(dtype=torch.float32)
    scale = amax / fp8_max
    return torch.clamp(scale, min=1e-8).contiguous()


def _quantize_weight_for_scaled_mm(
    weight_bf16: torch.Tensor,
    fp8_dtype: torch.dtype,
    fp8_scaling: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fp8_scaling == FP8_SCALING_PER_ROW:
        scale_w = _rowwise_scale(weight_bf16, fp8_dtype)  # [N, 1]
        weight_q = (weight_bf16 / scale_w).to(fp8_dtype)
        scale_b = scale_w.transpose(0, 1).contiguous()  # [1, N]
    elif fp8_scaling == FP8_SCALING_PER_TENSOR:
        scale_w = _tensorwise_scale(weight_bf16, fp8_dtype)
        weight_q = (weight_bf16 / scale_w).to(fp8_dtype)
        scale_b = scale_w
    else:
        raise ValueError(f"Unsupported FP8 scaling mode: {fp8_scaling}")

    # _scaled_mm expects rhs column-major (KxN with stride[0] == 1).
    return weight_q.t(), scale_b


class FP8LinearModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device,
        fp8_dtype: torch.dtype,
        fp8_scaling: str,
    ) -> None:
        super().__init__()
        if fp8_scaling not in {FP8_SCALING_PER_TENSOR, FP8_SCALING_PER_ROW}:
            raise ValueError(f"Unsupported FP8 scaling mode: {fp8_scaling}")

        self.fp8_dtype = fp8_dtype
        self.fp8_scaling = fp8_scaling

        weight_bf16 = torch.randn((out_features, in_features), device=device, dtype=torch.bfloat16)
        weight_q_t, scale_b = _quantize_weight_for_scaled_mm(
            weight_bf16=weight_bf16,
            fp8_dtype=fp8_dtype,
            fp8_scaling=fp8_scaling,
        )
        self.register_buffer("weight_q_t", weight_q_t)
        self.register_buffer("scale_b", scale_b)

        if bias:
            self.register_buffer("bias", torch.randn((out_features,), device=device, dtype=torch.bfloat16))
        else:
            self.bias = None

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fp8_scaling == FP8_SCALING_PER_ROW:
            scale_a = _rowwise_scale(x, self.fp8_dtype)  # [M, 1]
        else:
            scale_a = _tensorwise_scale(x, self.fp8_dtype)
        x_q = (x / scale_a).to(self.fp8_dtype)
        return x_q, scale_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, scale_a = self._quantize_activation(x)
        out = torch._scaled_mm(
            x_q,
            self.weight_q_t,
            scale_a=scale_a,
            scale_b=self.scale_b,
            out_dtype=torch.bfloat16,
        )
        out = out[0] if isinstance(out, tuple) else out
        if self.bias is not None:
            out = out + self.bias
        return out


def collect_shapes(args: argparse.Namespace) -> tuple[list[GemmShape], dict]:
    shapes: list[GemmShape] = []
    meta: dict = {}

    if "qwen" in args.families:
        qwen = derive_qwen_linear_gemm_shapes(
            height=args.qwen_height,
            width=args.qwen_width,
            prompt=args.qwen_prompt,
            prompt_lang=args.qwen_prompt_lang,
            negative_prompt=args.qwen_negative_prompt,
            max_sequence_length=args.qwen_max_sequence_length,
            prompt_seq_len=args.qwen_prompt_seq_len,
            negative_prompt_seq_len=args.qwen_negative_seq_len,
            include_negative_prompt_shapes=args.include_qwen_negative,
        )
        meta["qwen"] = qwen
        for item in qwen["shapes"]:
            shapes.append(
                GemmShape(
                    family="qwen",
                    name=item["name"],
                    m=item["m"],
                    k=item["k"],
                    n=item["n"],
                    occurrences=item["occurrences"],
                )
            )

    if "wan" in args.families:
        wan = derive_wan_linear_gemm_shapes(
            height=args.wan_height,
            width=args.wan_width,
            frames=args.wan_frames,
            text_seq_len=args.wan_text_seq_len,
        )
        meta["wan"] = wan
        for item in wan["shapes"]:
            shapes.append(
                GemmShape(
                    family="wan",
                    name=item["name"],
                    m=item["m"],
                    k=item["k"],
                    n=item["n"],
                    occurrences=item["occurrences"],
                )
            )

    if args.dedupe:
        merged: dict[tuple[int, int, int], GemmShape] = {}
        for shape in shapes:
            key = (shape.m, shape.k, shape.n)
            existing = merged.get(key)
            if existing is None:
                merged[key] = GemmShape(
                    family=shape.family,
                    name=shape.name,
                    m=shape.m,
                    k=shape.k,
                    n=shape.n,
                    occurrences=shape.occurrences,
                )
            else:
                merged[key] = GemmShape(
                    family=f"{existing.family}+{shape.family}" if shape.family not in existing.family else existing.family,
                    name=f"{existing.name} | {shape.name}",
                    m=shape.m,
                    k=shape.k,
                    n=shape.n,
                    occurrences=existing.occurrences + shape.occurrences,
                )
        shapes = list(merged.values())

    shapes.sort(key=lambda s: (s.m * s.n * s.k, s.m, s.k, s.n), reverse=True)
    if args.max_shapes is not None:
        shapes = shapes[: args.max_shapes]

    return shapes, meta


def _build_runner(
    shape: GemmShape,
    mode: str,
    bench_target: str,
    device: torch.device,
    compile_mode: str,
    compile_fullgraph: bool,
    linear_bias: bool,
    fp8_config: FP8Config,
):
    m, k, n = shape.m, shape.k, shape.n

    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)

    if bench_target == BENCH_TARGET_LINEAR:
        if mode == MODE_BF16:
            module = nn.Linear(k, n, bias=linear_bias, device=device, dtype=torch.bfloat16).eval()

            def run():
                return module(x)

            return run

        if mode == MODE_BF16_COMPILE:
            module = nn.Linear(k, n, bias=linear_bias, device=device, dtype=torch.bfloat16).eval()
            compiled = _compile_fn(module, mode=compile_mode, fullgraph=compile_fullgraph)

            def run():
                return compiled(x)

            return run

        if mode == MODE_FP8_COMPILE:
            module = FP8LinearModule(
                in_features=k,
                out_features=n,
                bias=linear_bias,
                device=device,
                fp8_dtype=fp8_config.dtype,
                fp8_scaling=fp8_config.scaling,
            ).eval()
            compiled = _compile_fn(module, mode=compile_mode, fullgraph=compile_fullgraph)

            def run():
                return compiled(x)

            return run

    if bench_target == BENCH_TARGET_GEMM:
        weight_t = torch.randn((k, n), device=device, dtype=torch.bfloat16)

        if mode == MODE_BF16:

            def run():
                return torch.matmul(x, weight_t)

            return run

        if mode == MODE_BF16_COMPILE:

            def gemm_fn(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
                return torch.matmul(a, b_t)

            compiled = _compile_fn(gemm_fn, mode=compile_mode, fullgraph=compile_fullgraph)

            def run():
                return compiled(x, weight_t)

            return run

        if mode == MODE_FP8_COMPILE:
            if fp8_config.scaling == FP8_SCALING_PER_ROW:
                scale_a = _rowwise_scale(x, fp8_config.dtype)
            else:
                scale_a = _tensorwise_scale(x, fp8_config.dtype)
            x_q = (x / scale_a).to(fp8_config.dtype)

            weight_bf16 = torch.randn((n, k), device=device, dtype=torch.bfloat16)
            weight_q_t, scale_b = _quantize_weight_for_scaled_mm(
                weight_bf16=weight_bf16,
                fp8_dtype=fp8_config.dtype,
                fp8_scaling=fp8_config.scaling,
            )

            def fp8_gemm_fn(
                a_q: torch.Tensor,
                b_q_t: torch.Tensor,
                scale_a_: torch.Tensor,
                scale_b_: torch.Tensor,
            ) -> torch.Tensor:
                out = torch._scaled_mm(
                    a_q,
                    b_q_t,
                    scale_a=scale_a_,
                    scale_b=scale_b_,
                    out_dtype=torch.bfloat16,
                )
                return out[0] if isinstance(out, tuple) else out

            compiled = _compile_fn(fp8_gemm_fn, mode=compile_mode, fullgraph=compile_fullgraph)

            def run():
                return compiled(x_q, weight_q_t, scale_a, scale_b)

            return run

    raise ValueError(f"Unsupported mode: {mode}")


def benchmark_with_cuda_events(run, warmup_iters: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup_iters):
        _ = run()
    torch.cuda.synchronize()

    elapsed_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = run()
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    elapsed_ms.sort()
    median_ms = statistics.median(elapsed_ms)
    mean_ms = sum(elapsed_ms) / len(elapsed_ms)
    p90_idx = min(len(elapsed_ms) - 1, int(round(0.9 * (len(elapsed_ms) - 1))))
    p90_ms = elapsed_ms[p90_idx]
    return median_ms, mean_ms, p90_ms


def format_shape(shape: GemmShape) -> str:
    return f"{shape.m}x{shape.k}x{shape.n}"


def print_shape_inventory(shapes: list[GemmShape], meta: dict) -> None:
    if "qwen" in meta:
        q = meta["qwen"]
        print(
            "[info] qwen: "
            f"{q['height']}x{q['width']} -> image_seq_len={q['image_seq_len']} "
            f"prompt_seq_len={q['prompt_seq_len']} negative_prompt_seq_len={q['negative_prompt_seq_len']}"
        )
    if "wan" in meta:
        w = meta["wan"]
        print(
            "[info] wan: "
            f"{w['height']}x{w['width']} frames={w['frames']} effective_frames={w['effective_frames']} "
            f"latent={w['latent_frames']}x{w['latent_height']}x{w['latent_width']} latent_seq_len={w['latent_seq_len']} "
            f"text_seq_len={w['text_seq_len']}"
        )

    print("[info] GEMM shape inventory:")
    for idx, shape in enumerate(shapes, start=1):
        print(
            f"  {idx:02d}. {shape.name} [{shape.family}] shape={format_shape(shape)} "
            f"occurrences={shape.occurrences}"
        )


def run_benchmarks(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got: {device}")

    if args.warmup_iters <= 0 or args.iters <= 0:
        raise ValueError("--warmup-iters and --iters must both be > 0")

    fp8_config = resolve_fp8_config(args.fp8_scaling)

    shapes, meta = collect_shapes(args)
    print_shape_inventory(shapes=shapes, meta=meta)
    if args.print_shapes_only:
        return 0

    print("[info] starting benchmarks")
    print(
        "[info] settings: "
        f"device={device} warmup_iters={args.warmup_iters} iters={args.iters} "
        f"bench_target={args.bench_target} "
        f"modes={args.modes} compile_mode={args.compile_mode} compile_fullgraph={args.compile_fullgraph} "
        f"linear_bias={args.linear_bias}"
    )
    print(
        "[info] fp8 config: "
        f"backend={fp8_config.backend} dtype={fp8_config.dtype} scaling={fp8_config.scaling}"
    )

    results: list[BenchResult] = []

    with torch.no_grad():
        for shape in shapes:
            for mode in args.modes:
                print(f"[run] {shape.name} mode={mode} shape={format_shape(shape)}")
                try:
                    run = _build_runner(
                        shape=shape,
                        mode=mode,
                        bench_target=args.bench_target,
                        device=device,
                        compile_mode=args.compile_mode,
                        compile_fullgraph=args.compile_fullgraph,
                        linear_bias=args.linear_bias,
                        fp8_config=fp8_config,
                    )
                    median_ms, mean_ms, p90_ms = benchmark_with_cuda_events(
                        run=run,
                        warmup_iters=args.warmup_iters,
                        iters=args.iters,
                    )
                    flops = 2.0 * float(shape.m) * float(shape.k) * float(shape.n)
                    tflops = flops / (median_ms * 1e-3) / 1e12
                    results.append(
                        BenchResult(
                            family=shape.family,
                            name=shape.name,
                            m=shape.m,
                            k=shape.k,
                            n=shape.n,
                            bench_target=args.bench_target,
                            mode=mode,
                            status="ok",
                            median_ms=median_ms,
                            mean_ms=mean_ms,
                            p90_ms=p90_ms,
                            tflops=tflops,
                            note="",
                        )
                    )
                    print(
                        f"[ok] {shape.name} mode={mode} median={median_ms:.4f} ms "
                        f"mean={mean_ms:.4f} ms p90={p90_ms:.4f} ms tflops={tflops:.2f}"
                    )
                except RuntimeError as exc:
                    lower = str(exc).lower()
                    if "out of memory" in lower:
                        note = "OOM"
                    elif "_scaled_mm" in lower and mode == MODE_FP8_COMPILE:
                        note = (
                            f"fp8 kernel/scaling unsupported "
                            f"(backend={fp8_config.backend}, scaling={fp8_config.scaling})"
                        )
                    else:
                        note = str(exc)
                    results.append(
                        BenchResult(
                            family=shape.family,
                            name=shape.name,
                            m=shape.m,
                            k=shape.k,
                            n=shape.n,
                            bench_target=args.bench_target,
                            mode=mode,
                            status="error",
                            median_ms=None,
                            mean_ms=None,
                            p90_ms=None,
                            tflops=None,
                            note=note,
                        )
                    )
                    print(f"[warn] {shape.name} mode={mode} failed: {note}")
                    torch.cuda.empty_cache()
                finally:
                    if hasattr(torch, "_dynamo"):
                        try:
                            torch._dynamo.reset()
                        except Exception:
                            pass

    print("\n[result] summary")
    header = f"{'shape':>20}  {'mode':>13}  {'median_ms':>10}  {'mean_ms':>10}  {'p90_ms':>10}  {'TFLOPs':>9}  status"
    print(header)
    print("-" * len(header))
    for row in results:
        shape_label = f"{row.m}x{row.k}x{row.n}"
        if row.status == "ok":
            print(
                f"{shape_label:>20}  {row.mode:>13}  {row.median_ms:10.4f}  {row.mean_ms:10.4f}  "
                f"{row.p90_ms:10.4f}  {row.tflops:9.2f}  {row.status}"
            )
        else:
            print(f"{shape_label:>20}  {row.mode:>13}  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>9}  {row.status}: {row.note}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "family",
                    "name",
                    "m",
                    "k",
                    "n",
                    "bench_target",
                    "mode",
                    "status",
                    "median_ms",
                    "mean_ms",
                    "p90_ms",
                    "tflops",
                    "note",
                ]
            )
            for row in results:
                writer.writerow(
                    [
                        row.family,
                        row.name,
                        row.m,
                        row.k,
                        row.n,
                        row.bench_target,
                        row.mode,
                        row.status,
                        row.median_ms,
                        row.mean_ms,
                        row.p90_ms,
                        row.tflops,
                        row.note,
                    ]
                )
        print(f"[ok] wrote csv to {csv_path}")

    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_benchmarks(args))


if __name__ == "__main__":
    main()
