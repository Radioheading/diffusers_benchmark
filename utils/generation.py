from typing import Any

import torch


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


def run_inference(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    true_cfg_scale: float,
    steps: int,
    height: int,
    width: int,
    seed: int | None,
    generator_device: str,
    run_device: torch.device,
):
    generator = make_generator(seed=seed, generator_device=generator_device, run_device=run_device)
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
