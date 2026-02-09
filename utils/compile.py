from collections.abc import Iterable
from typing import Any

import torch


def maybe_compile_pipeline(
    pipe: Any,
    enabled: bool,
    mode: str,
    fullgraph: bool = True,
    module_names: Iterable[str] = ("transformer", "unet"),
) -> None:
    if not enabled:
        print("[info] torch.compile disabled.")
        return

    if not hasattr(torch, "compile"):
        print("[warn] torch.compile is not available in this PyTorch build.")
        return

    for module_name in module_names:
        module = getattr(pipe, module_name, None)
        if module is None:
            continue

        try:
            compiled_module = torch.compile(module, mode=mode, fullgraph=fullgraph)
        except TypeError:
            # Older PyTorch versions may not support fullgraph and/or mode.
            try:
                compiled_module = torch.compile(module, mode=mode)
            except TypeError:
                compiled_module = torch.compile(module)
        except Exception as exc:
            print(f"[warn] torch.compile failed for {module_name}: {exc}")
            continue

        try:
            setattr(pipe, module_name, compiled_module)
            print(f"[ok] torch.compile enabled for {module_name}")
        except Exception as exc:
            print(f"[warn] torch.compile failed for {module_name}: {exc}")
