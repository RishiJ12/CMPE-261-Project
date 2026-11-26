"""Stable Diffusion 1.5 + ControlNet baseline from the working notebook."""

from pathlib import Path
import json
import time

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from .preprocessing import load_rgb, make_condition


def load_controlnet_pipeline(
    cfg: dict,
    device: str = "cuda",
):
    """Load the SD1.5 Canny ControlNet pipeline used by the baseline."""
    controlnet = ControlNetModel.from_pretrained(
        cfg["controlnet"],
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg["base_model"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_sequence(
    pipe,
    seq_name: str,
    panel_files: list[str],
    prompts: list[str],
    output_dir: str | Path,
    method: str = "controlnet_baseline",
    cfg: dict | None = None,
):
    """Generate a storyboard sequence using the ControlNet-only baseline."""
    if cfg is None:
        raise ValueError("cfg is required")

    assert len(panel_files) == len(prompts), "need one prompt per panel"

    out_dir = Path(output_dir) / method / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for index, (panel_file, prompt) in enumerate(
        zip(panel_files, prompts),
        1,
    ):
        condition = make_condition(
            load_rgb(panel_file),
            cfg["cond_mode"],
            cfg["resolution"],
            cfg["canny_low"],
            cfg["canny_high"],
        )

        generator = torch.Generator("cuda").manual_seed(cfg["seed"])

        image = pipe(
            prompt=f"{prompt}, {cfg['prompt_suffix']}",
            negative_prompt=cfg["negative_prompt"],
            image=condition,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"],
            controlnet_conditioning_scale=(
                cfg["controlnet_conditioning_scale"]
            ),
            generator=generator,
        ).images[0]

        image.save(out_dir / f"panel{index:02d}.png")
        condition.save(out_dir / f"panel{index:02d}_cond.png")
        results.append(image)

    record = {
        "method": method,
        "sequence": seq_name,
        "panel_files": panel_files,
        "prompts": prompts,
        "config": cfg,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
    }

    with open(out_dir / "run_config.json", "w") as file:
        json.dump(record, file, indent=2)

    return results
