"""ControlNet + IP-Adapter generation extracted from the working notebook."""

from pathlib import Path
import json
import time

import torch

from .preprocessing import load_rgb, make_condition


def attach_ip_adapter(pipe, ip_cfg: dict):
    """Attach the configured SD1.5 IP-Adapter to an existing pipeline."""
    pipe.load_ip_adapter(
        ip_cfg["repo"],
        subfolder=ip_cfg["subfolder"],
        weight_name=ip_cfg["weight_name"],
    )
    pipe.set_ip_adapter_scale(ip_cfg["scale"])

    if pipe.unet.encoder_hid_proj is None:
        raise RuntimeError(
            "IP-Adapter projection layers were not loaded."
        )

    if pipe.image_encoder is None:
        raise RuntimeError(
            "IP-Adapter image encoder was not loaded."
        )

    return pipe


def run_sequence_ip_adapter(
    pipe,
    seq_name: str,
    panel_files: list[str],
    prompts: list[str],
    character_reference,
    output_dir: str | Path,
    method: str = "controlnet_ip_adapter",
    cfg: dict | None = None,
    ip_cfg: dict | None = None,
    reference_path: str | Path | None = None,
):
    """
    Generate a storyboard sequence using ControlNet and IP-Adapter.

    ControlNet conditions storyboard structure while IP-Adapter conditions
    character appearance. Other generation settings match the baseline.
    """
    if cfg is None or ip_cfg is None:
        raise ValueError("cfg and ip_cfg are required")

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
            ip_adapter_image=character_reference,
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
        "ip_adapter": {
            "repo": ip_cfg["repo"],
            "subfolder": ip_cfg["subfolder"],
            "weight_name": ip_cfg["weight_name"],
            "scale": ip_cfg["scale"],
            "reference_image": (
                str(reference_path)
                if reference_path is not None
                else None
            ),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
    }

    with open(out_dir / "run_config.json", "w") as file:
        json.dump(record, file, indent=2)

    return results
