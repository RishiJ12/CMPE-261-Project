"""Storyboard structural-adherence metrics from the IP-Adapter notebook."""

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_score


def to_edge_mask(
    image: Image.Image,
    size: int = 512,
    low: int = 100,
    high: int = 200,
    dilation_pixels: int = 1,
) -> np.ndarray:
    """Convert an image to a binary Canny edge mask."""
    gray = image.convert("L").resize((size, size))

    edges = cv2.Canny(
        np.array(gray),
        low,
        high,
    )

    if dilation_pixels > 0:
        kernel_size = 2 * dilation_pixels + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )
        edges = cv2.dilate(
            edges,
            kernel,
            iterations=1,
        )

    return edges > 0


def compare_edge_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> dict[str, float]:
    """Compare two binary masks using Dice similarity and Jaccard/IoU."""
    a = mask_a.astype(bool).ravel()
    b = mask_b.astype(bool).ravel()

    dice_similarity = 1.0 - dice(a, b)

    jaccard = jaccard_score(
        a,
        b,
        zero_division=1,
    )

    return {
        "dice": float(dice_similarity),
        "jaccard": float(jaccard),
    }


def compare_structure(
    storyboard: Image.Image,
    generated: Image.Image,
    cfg: dict,
    dilation_pixels: int = 1,
) -> dict[str, float]:
    """Compare storyboard and generated structure after Canny extraction."""
    storyboard_edges = to_edge_mask(
        storyboard,
        size=cfg["resolution"],
        low=cfg["canny_low"],
        high=cfg["canny_high"],
        dilation_pixels=dilation_pixels,
    )

    generated_edges = to_edge_mask(
        generated,
        size=cfg["resolution"],
        low=cfg["canny_low"],
        high=cfg["canny_high"],
        dilation_pixels=dilation_pixels,
    )

    return compare_edge_masks(
        storyboard_edges,
        generated_edges,
    )
