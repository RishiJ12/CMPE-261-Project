"""
ConsiStory + SDXL ControlNet integration extracted from the working notebook.

Important:
- The official NVlabs/consistory repository must be cloned and importable.
- The module keeps the notebook's PyTorch-SDPA fallback for xFormers.
- DPM-Solver++ support uses a scheduler-aware DIFT store while preserving
  ConsiStory's legacy '261_0' lookup key.
"""

from pathlib import Path
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor

import consistory_run as cr
import consistory_utils
import attention_processor
from consistory_run import load_pipeline, run_anchor_generation
from consistory_utils import DIFTLatentStore

from .preprocessing import load_rgb, make_condition


class TorchSDPAOps:
    """xFormers-compatible wrapper around PyTorch scaled-dot-product attention."""

    @staticmethod
    def memory_efficient_attention(
        query,
        key,
        value,
        op=None,
        scale=None,
        **kwargs,
    ):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )


class TorchXFormersFallback:
    ops = TorchSDPAOps()


def install_torch_sdpa_fallback() -> TorchXFormersFallback:
    """Patch ConsiStory's xFormers references to use PyTorch SDPA."""
    fallback = TorchXFormersFallback()
    consistory_utils.xformers = fallback
    attention_processor.xformers = fallback
    return fallback


def load_consistory_pipeline(
    gpu_id: int = 0,
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = True,
):
    """Load the official ConsiStory SDXL pipeline and memory optimizations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipeline = load_pipeline(gpu_id)

    if enable_vae_slicing:
        pipeline.enable_vae_slicing()

    if enable_vae_tiling:
        pipeline.enable_vae_tiling()

    return pipeline


def load_consistory_controlnet(
    model_name: str,
    device: str = "cuda",
):
    """Load the SDXL Canny ControlNet used by the ConsiStory pipeline."""
    controlnet = ControlNetModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    controlnet.eval()
    return controlnet


def make_control_image_processor(story_pipeline) -> VaeImageProcessor:
    """Build the same RGB/no-normalization processor used by SDXL ControlNet."""
    return VaeImageProcessor(
        vae_scale_factor=story_pipeline.vae_scale_factor,
        do_convert_rgb=True,
        do_normalize=False,
    )


def prepare_control_image(
    condition,
    processor: VaeImageProcessor,
    resolution: int,
    device: str = "cuda",
    dtype=torch.float16,
):
    """Convert a PIL Canny condition to an SDXL ControlNet tensor."""
    tensor = processor.preprocess(
        condition,
        height=resolution,
        width=resolution,
    )
    return tensor.to(
        device=device,
        dtype=dtype,
    )


class ConsistoryControlNetUNet(nn.Module):
    """
    Transparent wrapper around the ConsiStory SDXL UNet.

    Unknown attributes delegate to the original ConsiStory UNet. During
    forward(), ControlNet residuals are injected into that UNet.
    """

    def __init__(
        self,
        unet,
        controlnet,
        conditioning_scale: float = 1.0,
    ):
        super().__init__()
        self.base_unet = unet
        self.controlnet = controlnet
        self.conditioning_scale = conditioning_scale
        self.control_image = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_unet = super().__getattr__("base_unet")
            return getattr(base_unet, name)

    def set_control_image(self, control_image):
        self.control_image = control_image

    def clear_control_image(self):
        self.control_image = None

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        timestep_cond=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        return_dict=True,
        **kwargs,
    ):
        if self.control_image is None:
            return self.base_unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=return_dict,
                **kwargs,
            )

        control_cond = self.control_image

        if control_cond.shape[0] != sample.shape[0]:
            assert (
                sample.shape[0] % control_cond.shape[0] == 0
            ), "Control image batch cannot be matched to latent batch"

            repeats = sample.shape[0] // control_cond.shape[0]
            control_cond = control_cond.repeat(
                repeats,
                1,
                1,
                1,
            )

        control_cond = control_cond.to(
            device=sample.device,
            dtype=sample.dtype,
        )

        (
            down_block_res_samples,
            mid_block_res_sample,
        ) = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_cond,
            conditioning_scale=self.conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        return self.base_unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=return_dict,
            **kwargs,
        )


def attach_controlnet(
    story_pipeline,
    controlnet,
    conditioning_scale: float = 1.0,
):
    """Wrap the ConsiStory UNet with the ControlNet bridge."""
    current_unet = story_pipeline.unet

    if isinstance(current_unet, ConsistoryControlNetUNet):
        base_unet = current_unet.base_unet
    elif hasattr(current_unet, "base_unet"):
        base_unet = current_unet.base_unet
    else:
        base_unet = current_unet

    story_pipeline.unet = ConsistoryControlNetUNet(
        unet=base_unet,
        controlnet=controlnet,
        conditioning_scale=conditioning_scale,
    )
    return story_pipeline


def generate_anchor_caches(
    story_pipeline,
    anchor_prompts: list[str],
    concept_token,
    cfg: dict,
    n_steps: int | None = None,
    cache_cpu_offloading: bool = True,
):
    """Generate ConsiStory identity anchors/caches using the active scheduler."""
    if hasattr(story_pipeline.unet, "clear_control_image"):
        story_pipeline.unet.clear_control_image()

    story_pipeline.unet.latent_store.reset()
    gc.collect()
    torch.cuda.empty_cache()

    steps = n_steps if n_steps is not None else cfg["steps"]

    return run_anchor_generation(
        story_pipeline=story_pipeline,
        prompts=anchor_prompts,
        concept_token=concept_token,
        seed=cfg["seed"],
        n_steps=steps,
        mask_dropout=cfg["mask_dropout"],
        same_latent=cfg["same_latent"],
        share_queries=cfg["share_queries"],
        perform_sdsa=True,
        perform_injection=True,
        cache_cpu_offloading=cache_cpu_offloading,
    )


def run_extra_generation_controlled(
    story_pipeline,
    prompts,
    concept_token,
    anchor_cache_first_stage,
    anchor_cache_second_stage,
    seed=42,
    n_steps=50,
    mask_dropout=0.5,
    same_latent=False,
    share_queries=True,
    perform_sdsa=True,
    perform_injection=True,
    cache_cpu_offloading=True,
    guidance_scale=7.5,
    negative_prompt=None,
):
    """
    Cached ConsiStory generation following the official helper while exposing
    benchmark CFG and negative-prompt controls.
    """
    device = story_pipeline.device
    tokenizer = story_pipeline.tokenizer
    float_type = story_pipeline.dtype
    unet = story_pipeline.unet

    batch_size = len(prompts)

    token_indices = cr.create_token_indices(
        prompts,
        batch_size,
        concept_token,
        tokenizer,
    )

    attention_store_kwargs = {
        "token_indices": token_indices,
        "mask_dropout": mask_dropout,
    }

    extended_attn_base = {
        "extend_kv_unet_parts": ["up"],
    }

    query_store_kwargs = {
        "t_range": [0, n_steps // 10],
        "strength_start": 0.9,
        "strength_end": 0.81836735,
    }

    extra_batch_size = batch_size + 2

    if isinstance(seed, list):
        seed = [seed[0], seed[0], *seed]

    latents, generator = cr.create_latents(
        story_pipeline,
        seed,
        extra_batch_size,
        same_latent,
        device,
        float_type,
    )
    latents = latents[2:]

    anchor_cache_first_stage.set_mode_inject()
    anchor_cache_second_stage.set_mode_inject()

    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(device)

    if perform_sdsa:
        extended_attn_kwargs = {
            **extended_attn_base,
            "t_range": [(1, n_steps)],
        }
    else:
        extended_attn_kwargs = {
            **extended_attn_base,
            "t_range": [],
        }

    out = story_pipeline(
        prompt=prompts,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        latents=latents,
        attention_store_kwargs=attention_store_kwargs,
        extended_attn_kwargs=extended_attn_kwargs,
        share_queries=share_queries,
        query_store_kwargs=query_store_kwargs,
        anchors_cache=anchor_cache_first_stage,
        num_inference_steps=n_steps,
    )

    last_masks = story_pipeline.attention_store.last_mask

    dift_features = (
        unet.latent_store.dift_features["261_0"][batch_size:]
    )

    dift_features = torch.stack(
        [
            cr.gaussian_smooth(
                feature,
                kernel_size=3,
                sigma=1,
            )
            for feature in dift_features
        ],
        dim=0,
    )

    anchor_dift_features = anchor_cache_first_stage.dift_cache
    anchor_last_masks = anchor_cache_first_stage.anchors_last_mask

    nn_map, nn_distances = cr.anchor_nn_map(
        dift_features,
        anchor_dift_features,
        last_masks,
        anchor_last_masks,
        cr.LATENT_RESOLUTIONS,
        device,
    )

    if cache_cpu_offloading:
        anchor_cache_first_stage.to_device(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    if perform_injection:
        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(device)

        feature_injector = cr.FeatureInjector(
            nn_map,
            nn_distances,
            last_masks,
            inject_range_alpha=[
                (
                    n_steps // 10,
                    n_steps // 3,
                    0.8,
                )
            ],
            swap_strategy="min",
            inject_unet_parts=["up", "down"],
            dist_thr="dynamic",
        )

        out = story_pipeline(
            prompt=prompts,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            attention_store_kwargs=attention_store_kwargs,
            extended_attn_kwargs=extended_attn_kwargs,
            share_queries=share_queries,
            query_store_kwargs=query_store_kwargs,
            feature_injector=feature_injector,
            anchors_cache=anchor_cache_second_stage,
            num_inference_steps=n_steps,
        )

        if cache_cpu_offloading:
            anchor_cache_second_stage.to_device(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    return out.images


def run_sequence_consistory(
    story_pipeline,
    control_image_processor,
    seq_name: str,
    panel_files: list[str],
    prompts: list[str],
    anchor_cache_first_stage,
    anchor_cache_second_stage,
    output_dir: str | Path,
    concept_token,
    anchor_prompts: list[str],
    method: str = "controlnet_consistory",
    cfg: dict | None = None,
    n_steps: int | None = None,
):
    """Generate and save a complete ControlNet + ConsiStory storyboard sequence."""
    if cfg is None:
        raise ValueError("cfg is required")

    assert len(panel_files) == len(prompts), "need one prompt per panel"

    out_dir = Path(output_dir) / method / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = n_steps if n_steps is not None else cfg["steps"]
    results = []
    runtimes = []

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

        control_tensor = prepare_control_image(
            condition,
            processor=control_image_processor,
            resolution=cfg["resolution"],
        )

        story_pipeline.unet.set_control_image(control_tensor)
        story_pipeline.unet.conditioning_scale = (
            cfg["controlnet_conditioning_scale"]
        )
        story_pipeline.unet.latent_store.reset()

        gc.collect()
        torch.cuda.empty_cache()

        full_prompt = f"{prompt}, {cfg['prompt_suffix']}"
        start = time.perf_counter()

        try:
            images = run_extra_generation_controlled(
                story_pipeline=story_pipeline,
                prompts=[full_prompt],
                concept_token=concept_token,
                anchor_cache_first_stage=anchor_cache_first_stage,
                anchor_cache_second_stage=anchor_cache_second_stage,
                seed=cfg["seed"],
                n_steps=steps,
                mask_dropout=cfg["mask_dropout"],
                same_latent=cfg["same_latent"],
                share_queries=cfg["share_queries"],
                perform_sdsa=True,
                perform_injection=True,
                cache_cpu_offloading=True,
                guidance_scale=cfg["guidance_scale"],
                negative_prompt=cfg["negative_prompt"],
            )

            elapsed = time.perf_counter() - start
            image = images[0]

            image.save(out_dir / f"panel{index:02d}.png")
            condition.save(out_dir / f"panel{index:02d}_cond.png")

            results.append(image)
            runtimes.append(elapsed)

        finally:
            story_pipeline.unet.clear_control_image()
            del control_tensor
            gc.collect()
            torch.cuda.empty_cache()

    record = {
        "method": method,
        "sequence": seq_name,
        "panel_files": panel_files,
        "prompts": prompts,
        "config": cfg,
        "steps_used": steps,
        "consistory": {
            "concept_token": concept_token,
            "anchor_prompts": anchor_prompts,
            "perform_sdsa": True,
            "perform_injection": True,
            "cache_cpu_offloading": True,
            "attention_backend": "torch_sdpa",
        },
        "runtime_sec_per_panel": runtimes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
    }

    with open(out_dir / "run_config.json", "w") as file:
        json.dump(record, file, indent=2)

    return results


class SchedulerAwareDIFTLatentStore(DIFTLatentStore):
    """
    Capture DIFT features at the active scheduler timestep and expose a
    '261_0' compatibility alias expected by the released ConsiStory helper.
    """

    def __init__(
        self,
        capture_timestep,
        up_ft_indices=(0,),
    ):
        super().__init__(
            steps=[int(capture_timestep)],
            up_ft_indices=list(up_ft_indices),
        )
        self.capture_timestep = int(capture_timestep)

    def __call__(
        self,
        features,
        t,
        layer_index,
    ):
        t_int = int(t)

        if (
            t_int == self.capture_timestep
            and layer_index in self.up_ft_indices
        ):
            self.dift_features[
                f"{t_int}_{layer_index}"
            ] = features

            self.dift_features[
                f"261_{layer_index}"
            ] = features


def get_base_consistory_unet(story_pipeline):
    """Return the underlying ConsiStory UNet whether or not it is wrapped."""
    unet = story_pipeline.unet
    return getattr(unet, "base_unet", unet)


def configure_dpm_solver_pp(
    story_pipeline,
    n_steps: int = 20,
    original_dift_timestep: int = 261,
):
    """
    Switch the active ConsiStory scheduler to second-order DPM-Solver++ and
    configure scheduler-aware DIFT capture.

    Returns the chosen DIFT capture timestep.
    """
    story_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        story_pipeline.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
        solver_type="midpoint",
        thresholding=False,
    )

    story_pipeline.scheduler.set_timesteps(n_steps)

    timesteps = [
        int(timestep)
        for timestep in story_pipeline.scheduler.timesteps
    ]

    capture_timestep = min(
        timesteps,
        key=lambda timestep: abs(
            timestep - original_dift_timestep
        ),
    )

    base_unet = get_base_consistory_unet(story_pipeline)
    base_unet.latent_store = SchedulerAwareDIFTLatentStore(
        capture_timestep=capture_timestep,
        up_ft_indices=(0,),
    )

    return capture_timestep


def run_single_panel_consistory(
    panel_index: int,
    story_pipeline,
    control_image_processor,
    panel_files: list[str],
    prompts: list[str],
    anchor_cache_first_stage,
    anchor_cache_second_stage,
    concept_token,
    cfg: dict,
    n_steps: int,
):
    """Run one storyboard panel; useful for pilots and scheduler comparisons."""
    panel_file = panel_files[panel_index]
    storyboard = load_rgb(panel_file)

    condition = make_condition(
        storyboard,
        mode=cfg["cond_mode"],
        size=cfg["resolution"],
        low=cfg["canny_low"],
        high=cfg["canny_high"],
    )

    control_tensor = prepare_control_image(
        condition,
        processor=control_image_processor,
        resolution=cfg["resolution"],
    )

    full_prompt = f"{prompts[panel_index]}, {cfg['prompt_suffix']}"

    story_pipeline.unet.set_control_image(control_tensor)
    story_pipeline.unet.conditioning_scale = (
        cfg["controlnet_conditioning_scale"]
    )

    get_base_consistory_unet(story_pipeline).latent_store.reset()

    gc.collect()
    torch.cuda.empty_cache()

    start = time.perf_counter()

    try:
        images = run_extra_generation_controlled(
            story_pipeline=story_pipeline,
            prompts=[full_prompt],
            concept_token=concept_token,
            anchor_cache_first_stage=anchor_cache_first_stage,
            anchor_cache_second_stage=anchor_cache_second_stage,
            seed=cfg["seed"],
            n_steps=n_steps,
            mask_dropout=cfg["mask_dropout"],
            same_latent=cfg["same_latent"],
            share_queries=cfg["share_queries"],
            perform_sdsa=True,
            perform_injection=True,
            cache_cpu_offloading=True,
            guidance_scale=cfg["guidance_scale"],
            negative_prompt=cfg["negative_prompt"],
        )

        elapsed = time.perf_counter() - start
        output = images[0]

    finally:
        story_pipeline.unet.clear_control_image()
        del control_tensor
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "panel_index": panel_index,
        "panel_name": f"Panel {panel_index + 1}",
        "panel_file": panel_file,
        "prompt": prompts[panel_index],
        "storyboard": storyboard,
        "condition": condition,
        "output": output,
        "time_sec": elapsed,
    }
