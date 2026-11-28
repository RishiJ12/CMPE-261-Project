"""Consistent Self-Attention for storyboard-conditioned ControlNet generation.

Adapted from the StoryDiffusion-style attention implementation in the
storyboard ControlNet notebook. This is not the full StoryDiffusion pipeline.
"""

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from PIL import Image

from .preprocessing import load_rgb, make_condition


class ConsistentSelfAttnProcessor:
    """Share sampled self-attention tokens between panels in one batch."""

    def __init__(
        self,
        enabled: bool = False,
        groups: int = 2,
        rate: float = 0.5,
        max_tokens: int = 4096
    ) -> None:
        self.enabled = enabled
        self.groups = groups
        self.rate = rate
        self.max_tokens = max_tokens
        self.default = AttnProcessor2_0()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        if not self.enabled or encoder_hidden_states is not None:
            return self.default(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, temb, *args, **kwargs
            )

        batch_size, num_tokens, channels = hidden_states.shape
        if batch_size % self.groups != 0:
            raise ValueError("Attention batch size must be divisible by groups")

        panels_per_group = batch_size // self.groups
        if panels_per_group < 2 or num_tokens > self.max_tokens:
            return self.default(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, temb, *args, **kwargs
            )

        features = hidden_states.reshape(
            self.groups, panels_per_group, num_tokens, channels
        )
        contexts = []

        for panel_index in range(panels_per_group):
            other_features = torch.cat(
                [
                    features[:, index]
                    for index in range(panels_per_group)
                    if index != panel_index
                ],
                dim=1
            )
            sampled_count = int(self.rate * other_features.shape[1])
            indices = torch.randperm(
                other_features.shape[1], device=features.device
            )[:sampled_count]
            contexts.append(
                torch.cat(
                    [features[:, panel_index], other_features[:, indices]],
                    dim=1
                )
            )

        context = torch.stack(contexts, dim=1).reshape(
            batch_size, num_tokens + sampled_count, channels
        )
        return self.default(
            attn, hidden_states, context, attention_mask, temb,
            *args, **kwargs
        )


def install_csa(pipe: StableDiffusionControlNetPipeline) -> int:
    """Replace UNet self-attention processors with CSA processors."""
    processors = {}
    count = 0

    for name, processor in pipe.unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            processors[name] = ConsistentSelfAttnProcessor()
            count += 1
        else:
            processors[name] = processor

    if count == 0:
        raise RuntimeError("No self-attention processors found in the UNet")

    pipe.unet.set_attn_processor(processors)
    return count


def warn_black(images: Sequence[Image.Image], label: str) -> None:
    """Report fully black generated images for manual inspection."""
    for index, image in enumerate(images, 1):
        if np.asarray(image).max() < 5:
            print(f"Warning: {label} panel {index} is all black")


def run_sequence_batch(
    pipe: StableDiffusionControlNetPipeline,
    seq_name: str,
    panel_files: Sequence[str | Path],
    prompts: Sequence[str],
    output_dir: str | Path,
    method: str = "controlnet_storydiffusion_csa",
    use_csa: bool = True,
    cfg: dict[str, Any] | None = None,
    rate: float = 0.5,
    max_tokens: int = 4096
) -> list[Image.Image]:
    """Generate a storyboard sequence in one batch using optional CSA."""
    if cfg is None:
        raise ValueError("cfg is required")
    if not panel_files or len(panel_files) != len(prompts):
        raise ValueError("Expected one prompt per storyboard panel")
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    processors = [
        processor for processor in pipe.unet.attn_processors.values()
        if isinstance(processor, ConsistentSelfAttnProcessor)
    ]
    if not processors:
        raise RuntimeError("Call install_csa(pipe) before run_sequence_batch")

    groups = 2 if cfg["guidance_scale"] > 1 else 1
    previous_settings = [
        (processor, processor.enabled, processor.groups, processor.rate, processor.max_tokens)
        for processor in processors
    ]

    conditions = [
        make_condition(
            load_rgb(path),
            cfg["cond_mode"],
            cfg["resolution"],
            cfg["canny_low"],
            cfg["canny_high"]
        )
        for path in panel_files
    ]

    # The seed controls attention sampling, and each panel starts from the same noise seed.
    torch.manual_seed(cfg["seed"])
    generators = [
        torch.Generator(device=str(pipe.device)).manual_seed(cfg["seed"])
        for _ in panel_files
    ]

    for processor in processors:
        processor.enabled = use_csa
        processor.groups = groups
        processor.rate = rate
        processor.max_tokens = max_tokens

    try:
        start = time.perf_counter()
        images = pipe(
            prompt=[f"{prompt}, {cfg['prompt_suffix']}" for prompt in prompts],
            negative_prompt=[cfg["negative_prompt"]] * len(prompts),
            image=conditions,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"],
            controlnet_conditioning_scale=cfg["controlnet_conditioning_scale"],
            generator=generators
        ).images
        runtime_sec = time.perf_counter() - start
    finally:
        for processor, enabled, old_groups, old_rate, old_max_tokens in previous_settings:
            processor.enabled = enabled
            processor.groups = old_groups
            processor.rate = old_rate
            processor.max_tokens = old_max_tokens

    warn_black(images, method)

    save_dir = Path(output_dir) / method / seq_name
    save_dir.mkdir(parents=True, exist_ok=True)
    for index, (image, condition) in enumerate(zip(images, conditions), 1):
        image.save(save_dir / f"panel{index:02d}.png")
        condition.save(save_dir / f"panel{index:02d}_cond.png")

    record = {
        "method": method,
        "sequence": seq_name,
        "panel_files": [str(path) for path in panel_files],
        "prompts": list(prompts),
        "config": cfg,
        "csa": {
            "enabled": use_csa,
            "groups": groups,
            "rate": rate,
            "max_tokens": max_tokens
        },
        "runtime_sec": runtime_sec,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__
    }
    with open(save_dir / "run_config.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)

    return list(images)
