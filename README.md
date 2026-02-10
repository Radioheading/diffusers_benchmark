# diffusers_benchmark

Benchmark scripts for Qwen-Image and Wan2.2 on local hardware.

## Script paths

- Generation benchmarks:
  - `tests/gen/benchmark_qwen.py`
  - `tests/gen/benchmark_wan.py`
- Operator microbenchmark:
  - `tests/op/benchmark_gemm.py`

## Model coverage

- `Qwen/Qwen-Image` via `tests/gen/benchmark_qwen.py`
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` via `tests/gen/benchmark_wan.py`

## Shared features

- Device selection: `--device cpu`, `--device cuda`, `--device cuda:0`, `--device gpu`, `--device gpu:0`
- Optional `torch.compile`: `--compile` / `--no-compile` plus `--compile-mode ...`
- CPU execution for both generation scripts
- Optional FP8 loading for generation scripts via `--quantization fp8_e4m3`
- Timing:
  - generation scripts run 1 warmup pass + 1 measured pass
  - `tests/op/benchmark_gemm.py` uses CUDA event timing with configurable warmup/iters

## Quick usage

Qwen (BF16 GPU):

```bash
python tests/gen/benchmark_qwen.py --device gpu --no-compile --output outputs/test_qwen/gpu.png
```

Qwen (FP8 GPU):

```bash
python tests/gen/benchmark_qwen.py --device gpu --no-compile --quantization fp8_e4m3 --fp8-quant-type float8dq_e4m3 --output outputs/test_qwen/gpu_fp8.png
```

Qwen (CPU):

```bash
python tests/gen/benchmark_qwen.py --device cpu --no-compile --output outputs/test_qwen/cpu.png
```

Wan2.2 (GPU):

```bash
python tests/gen/benchmark_wan.py --device gpu --output outputs/test_wan/wan2_2.mp4
```

Wan2.2 (CPU):

```bash
python tests/gen/benchmark_wan.py --device cpu --no-compile --output outputs/test_wan/wan2_2_cpu.mp4
```

## GEMM/Linear microbenchmark

`tests/op/benchmark_gemm.py` benchmarks representative `M x K x N` shapes derived from Qwen/Wan configs.

- Requires CUDA (`--device cuda`/`cuda:0`)
- Default modes: `bf16`, `bf16_compile`, `fp8_compile`
- Shape sources are selectable with `--families qwen wan`

List the shape inventory only:

```bash
python tests/op/benchmark_gemm.py --print-shapes-only
```

Run full Linear-module forward path (default target):

```bash
python tests/op/benchmark_gemm.py --device cuda --bench-target linear --csv outputs/gemm/linear.csv
```

Run kernel-focused GEMM path:

```bash
python tests/op/benchmark_gemm.py --device cuda --bench-target gemm --csv outputs/gemm/gemm.csv
```

Benchmark target behavior:

- `--bench-target linear`:
  - BF16/BF16+compile use `nn.Linear`
  - FP8+compile uses the FP8 linear wrapper
  - supports `--linear-bias` / `--no-linear-bias`
- `--bench-target gemm`:
  - BF16/BF16+compile use `torch.matmul`
  - FP8+compile uses `torch._scaled_mm`
  - ignores `--linear-bias`

FP8 policy when `--fp8-scaling auto`:

- AMD ROCm: `float8_e4m3fnuz` + `per_tensor`
- NVIDIA CUDA: `float8_e4m3fn` + `per_row`
