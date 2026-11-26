"""Character-reference extraction from the IP-Adapter notebook."""

from pathlib import Path
from typing import Protocol, TypeAlias

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Florence2ForConditionalGeneration,
    Sam2Model,
    Sam2Processor,
)

BBox: TypeAlias = tuple[int, int, int, int]


class CharacterLocator(Protocol):
    """Locates a requested character within an image."""

    def locate(self, image: Image.Image, prompt: str) -> BBox:
        ...


class CharacterSegmenter(Protocol):
    """Segments a character using a supplied bounding box."""

    def segment(self, image: Image.Image, bbox: BBox) -> Image.Image:
        ...


class BackgroundStrategy(Protocol):
    """Defines how the segmented character background is handled."""

    def apply(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        ...


class SolidBackgroundStrategy:
    """Replace the background with a solid color."""

    def __init__(
        self,
        color: str | tuple[int, int, int] = "white",
    ) -> None:
        self.color = color

    def apply(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        image = image.convert("RGB")
        background = Image.new(
            mode="RGB",
            size=image.size,
            color=self.color,
        )
        return Image.composite(image, background, mask)


def pad_bbox(
    bbox: BBox,
    image_size: tuple[int, int],
    padding: int = 10,
) -> BBox:
    """Add padding around a box without exceeding image bounds."""
    left, top, right, bottom = bbox
    width, height = image_size

    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


class Florence2CharacterLocator:
    """Use Florence-2 caption-to-phrase grounding to locate a character."""

    TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"

    def __init__(
        self,
        model_name: str = "florence-community/Florence-2-base",
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            Florence2ForConditionalGeneration
            .from_pretrained(
                model_name,
                dtype=self.dtype,
            )
            .to(self.device)
        )
        self.model.eval()

    @staticmethod
    def _bbox_area(bbox: BBox) -> int:
        left, top, right, bottom = bbox
        return max(0, right - left) * max(0, bottom - top)

    def _select_bbox(self, parsed_output: dict) -> BBox:
        task_result = parsed_output.get(self.TASK_PROMPT)

        if task_result is None:
            raise ValueError(
                f"Florence-2 did not return results for "
                f"{self.TASK_PROMPT}.\nOutput: {parsed_output}"
            )

        bboxes = task_result.get("bboxes", [])
        if not bboxes:
            raise ValueError(
                "Florence-2 did not detect a matching character."
            )

        bboxes = [tuple(map(int, bbox)) for bbox in bboxes]
        return max(bboxes, key=self._bbox_area)

    def locate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> BBox:
        image = image.convert("RGB")
        model_prompt = self.TASK_PROMPT + prompt

        inputs = self.processor(
            text=model_prompt,
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: (
                value.to(self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        parsed_output = self.processor.post_process_generation(
            generated_text,
            task=self.TASK_PROMPT,
            image_size=image.size,
        )

        return self._select_bbox(parsed_output)


class SAM2CharacterSegmenter:
    """Segment a character using SAM 2.1 and a bounding-box prompt."""

    def __init__(
        self,
        model_name: str = "facebook/sam2.1-hiera-tiny",
        device: str | None = None,
    ) -> None:
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = Sam2Processor.from_pretrained(model_name)
        self.model = Sam2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def segment(
        self,
        image: Image.Image,
        bbox: BBox,
    ) -> Image.Image:
        image = image.convert("RGB")
        input_boxes = [[list(map(float, bbox))]]

        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                multimask_output=False,
            )

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        mask_tensor = masks[0, 0]
        mask_array = mask_tensor.to(torch.uint8).numpy() * 255

        return Image.fromarray(mask_array, mode="L")


class CharacterReferenceExtractor:
    """Locate, segment, isolate, and tightly crop a character."""

    def __init__(
        self,
        locator: CharacterLocator,
        segmenter: CharacterSegmenter,
        background_strategy: BackgroundStrategy,
        padding: int = 10,
    ) -> None:
        self.locator = locator
        self.segmenter = segmenter
        self.background_strategy = background_strategy
        self.padding = padding

    def extract(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Image.Image:
        image = image.convert("RGB")

        bbox = self.locator.locate(
            image=image,
            prompt=prompt,
        )
        mask = self.segmenter.segment(
            image=image,
            bbox=bbox,
        )

        mask_bbox = mask.getbbox()
        if mask_bbox is None:
            raise ValueError(
                f"SAM 2 returned an empty mask for prompt: {prompt!r}"
            )

        crop_bbox = pad_bbox(
            bbox=mask_bbox,
            image_size=image.size,
            padding=self.padding,
        )

        isolated_character = self.background_strategy.apply(
            image=image,
            mask=mask,
        )

        return isolated_character.crop(crop_bbox)

    def extract_from_file(
        self,
        image_path: str | Path,
        prompt: str,
    ) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return self.extract(image=image, prompt=prompt)

    def extract_and_save(
        self,
        image_path: str | Path,
        output_path: str | Path,
        prompt: str,
    ) -> Image.Image:
        character_ref = self.extract_from_file(
            image_path=image_path,
            prompt=prompt,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        character_ref.save(output_path)

        return character_ref


def load_character_reference(path: str | Path) -> Image.Image:
    """Load a saved character reference as RGB."""
    return Image.open(path).convert("RGB")
