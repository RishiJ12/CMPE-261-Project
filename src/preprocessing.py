"""Shared storyboard loading, panel extraction, visualization, and conditioning."""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_rgb(path: str | Path) -> Image.Image:
    """Load an image as RGB, dropping alpha channels that can break downstream code."""
    return Image.open(path).convert("RGB")


def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    first_page: int = 9,
    last_page: int | None = 11,
    dpi: int = 200,
) -> list[Path]:
    """Render a page range from the storyboard PDF to PNGs using PyMuPDF."""
    import pymupdf

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    last = last_page or len(doc)
    zoom = dpi / 72

    paths: list[Path] = []
    for pno in range(first_page - 1, last):
        pix = doc[pno].get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            alpha=False,
        )
        path = output_dir / f"page{pno + 1:03d}.png"
        pix.save(path)
        paths.append(path)

    return paths


def split_grid(
    img: Image.Image,
    rows: int,
    cols: int,
    margin: int = 0,
) -> list[Image.Image]:
    """Evenly split a page into rows x cols panels in reading order."""
    width, height = img.size
    panel_width = width // cols
    panel_height = height // rows

    panels = []
    for row in range(rows):
        for col in range(cols):
            box = (
                col * panel_width + margin,
                row * panel_height + margin,
                (col + 1) * panel_width - margin,
                (row + 1) * panel_height - margin,
            )
            panels.append(img.crop(box))

    return panels


def detect_panels(
    img: Image.Image,
    min_w: int = 150,
    min_h: int = 100,
    thresh_val: int = 200,
    row_tol: int = 50,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    """Rough panel auto-detection using visible borders/contours."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(
        gray,
        thresh_val,
        255,
        cv2.THRESH_BINARY_INV,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_w and h > min_h:
            boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda box: (box[1] // row_tol, box[0]))
    return [img.crop(box) for box in boxes], boxes


def show(
    images: Iterable[Image.Image],
    titles: list[str] | None = None,
    cols: int = 4,
    size: int = 4,
) -> None:
    """Display a list of images in a simple matplotlib grid."""
    images = list(images)
    rows = (len(images) + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * size, rows * size),
    )
    axes = np.array(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for index, image in enumerate(images):
        axes[index].imshow(image)
        if titles:
            axes[index].set_title(titles[index], fontsize=9)

    plt.tight_layout()
    plt.show()


def save_sequence(
    panel_images: Iterable[Image.Image],
    seq_name: str,
    panel_dir: str | Path,
) -> list[str]:
    """Save panels with the shared seq##_panel###.png naming convention."""
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for index, panel in enumerate(panel_images, 1):
        path = panel_dir / f"{seq_name}_panel{index:03d}.png"
        panel.save(path)
        paths.append(str(path))

    return paths


def make_condition(
    img: Image.Image,
    mode: str = "canny",
    size: int = 512,
    low: int = 100,
    high: int = 200,
) -> Image.Image:
    """Create the shared Canny or inverted structural condition."""
    img = img.resize((size, size))
    gray = np.array(img.convert("L"))

    if mode == "canny":
        condition = cv2.Canny(gray, low, high)
    elif mode == "invert":
        condition = 255 - gray
    else:
        raise ValueError(mode)

    return Image.fromarray(condition).convert("RGB")
