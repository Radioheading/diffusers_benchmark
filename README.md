# test_speed

Small benchmark scripts for Qwen-Image and Wan2.2 on local hardware.

## Supported models

- `Qwen/Qwen-Image` via `benchmark_qwen.py`
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` via `benchmark_wan.py`

## Utilities and features

- Device selection:
  - `--device cpu`
  - `--device cuda` / `--device cuda:0`
  - `--device gpu` / `--device gpu:0` (alias)
- Optional `torch.compile`:
  - `--compile` (default)
  - `--no-compile`
  - `--compile-mode ...`
- CPU run support:
  - both benchmark scripts support CPU mode
- FP8 quantization (Qwen only):
  - `--quantization fp8_e4m3`
  - `--fp8-quant-type float8dq_e4m3` or `float8dq_e4m3_row`
- Speed measurement:
  - both scripts run one warmup pass and one measured pass
  - measured pass time is printed as:
    - `[timing] measured inference: ...`

## Quick usage

Qwen (BF16 baseline on GPU):

```bash
python benchmark_qwen.py --device gpu --no-compile --output outputs/test_qwen/gpu.png
```

Qwen (FP8):

```bash
python benchmark_qwen.py --device gpu --no-compile --quantization fp8_e4m3 --fp8-quant-type float8dq_e4m3 --output outputs/test_qwen/gpu_fp8.png
```

Qwen (CPU):

```bash
python benchmark_qwen.py --device cpu --no-compile --output outputs/test_qwen/cpu.png
```

Wan2.2 (GPU):

```bash
python benchmark_wan.py --device gpu --output outputs/test_wan/wan2_2.mp4
```

Wan2.2 (CPU):

```bash
python benchmark_wan.py --device cpu --no-compile --output outputs/test_wan/wan2_2_cpu.mp4
```
