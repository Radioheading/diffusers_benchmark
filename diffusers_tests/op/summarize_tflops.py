import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpSpec:
    label: str
    csv_name: str


OP_SPECS = (
    OpSpec("wan_ffn_out", "wan.ffn_out"),
    OpSpec("wan_ffn_in", "wan.ffn_in"),
    OpSpec(
        "wan_self_attn_qkvo",
        "wan.self_attn_qkv | wan.self_attn_out | wan.cross_attn_q | wan.cross_attn_out",
    ),
    OpSpec("qwen_img_ffn_out", "qwen.img_ffn_out"),
    OpSpec("qwen_img_ffn_in", "qwen.img_ffn_in"),
    OpSpec("qwen_img_attn_qkvo", "qwen.img_attn_qkv | qwen.img_attn_out"),
)

MODE_ORDER = ("bf16", "bf16_compile", "fp8_compile")
MODE_LABELS = {
    "bf16": "bf16",
    "bf16_compile": "w. compile",
    "fp8_compile": "fp8+compile",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def extract_op_modes(rows: list[dict[str, str]], op_name: str) -> tuple[str | None, dict[str, str]]:
    shape = None
    values: dict[str, str] = {}
    for row in rows:
        if row.get("name") != op_name:
            continue
        if shape is None:
            shape = f"{row['m']}×{row['k']}×{row['n']}"
        mode = row.get("mode", "")
        if mode in MODE_ORDER:
            values[mode] = row.get("tflops", "")
    return shape, values


def build_summary(csv_path: Path, title: str, strict: bool) -> tuple[list[str], list[str]]:
    rows = load_csv_rows(csv_path)
    lines = [f"[{title}]", ""]
    missing: list[str] = []

    for spec in OP_SPECS:
        shape, values = extract_op_modes(rows, spec.csv_name)
        lines.append(spec.label)

        if not shape:
            missing.append(f"{title}:{spec.label}: missing operator row ({spec.csv_name})")
            lines.append("shape: <missing>, bf16, <missing>")
            lines.append("shape: <missing>, w. compile, <missing>")
            lines.append("shape: <missing>, fp8+compile, <missing>")
            lines.append("")
            continue

        for mode in MODE_ORDER:
            val = values.get(mode, "")
            if not val:
                missing.append(f"{title}:{spec.label}: missing mode {mode}")
                val = "<missing>"
            lines.append(f"shape: {shape}, {MODE_LABELS[mode]}, {val}")
        lines.append("")

    if strict and missing:
        lines.append("[ERROR] missing entries detected:")
        lines.extend(missing)

    return lines, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize GEMM/Linear TFLOPs from smoke CSVs with strict 3-mode checks."
    )
    parser.add_argument(
        "--gemm-csv",
        type=Path,
        default=Path("outputs/gemm/smoke_module_gpu2_gemm.csv"),
    )
    parser.add_argument(
        "--linear-csv",
        type=Path,
        default=Path("outputs/gemm/smoke_module_gpu2_linear.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output text file path",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="do not return non-zero exit code when entries are missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strict = not args.no_strict

    all_lines: list[str] = []
    all_missing: list[str] = []

    for title, csv_path in (("GEMM", args.gemm_csv), ("LINEAR", args.linear_csv)):
        if not csv_path.exists():
            msg = f"[ERROR] csv not found: {csv_path}"
            print(msg)
            return 2

        lines, missing = build_summary(csv_path=csv_path, title=title, strict=strict)
        all_lines.extend(lines)
        all_missing.extend(missing)

    output_text = "\n".join(all_lines).rstrip() + "\n"
    print(output_text, end="")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_text, encoding="utf-8")

    if strict and all_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
