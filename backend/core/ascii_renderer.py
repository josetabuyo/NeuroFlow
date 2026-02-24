"""Image renderer for visual input streams.

Renders characters (or any drawable) into binary pixel grids using real
TrueType fonts via Pillow. Supports a registry of bundled and system fonts
with configurable size. Designed for low-resolution grids (10x10 and up).

Bundled fonts (OFL licensed):
  - Press Start 2P: pixel font from 1980s arcade games, best at 8pt
  - Silkscreen: pixel font designed for small screen rendering
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError(
        "Pillow >= 10.1 is required for rendering: pip install Pillow"
    )

_FONTS_DIR = Path(__file__).parent / "fonts"
_RENDER_SCALE = 20
_THRESHOLD = 0.35
_DEFAULT_NOISE_PROB = 0.15

# ── Font registry ──────────────────────────────────────────────────────
# Each entry: (display_name, path_or_None, recommended_sizes, description)
# path=None means system font or PIL default.

_BUNDLED_FONTS: dict[str, dict] = {
    "press_start_2p": {
        "name": "Press Start 2P",
        "file": "PressStart2P-Regular.ttf",
        "sizes": [7, 8, 9, 10, 12, 14],
        "default_size": 8,
        "description": "Pixel font — arcade style, best at 8pt multiples",
    },
    "silkscreen": {
        "name": "Silkscreen",
        "file": "Silkscreen-Regular.ttf",
        "sizes": [7, 8, 9, 10, 12, 14],
        "default_size": 10,
        "description": "Pixel font — designed for small screen rendering",
    },
    "silkscreen_bold": {
        "name": "Silkscreen Bold",
        "file": "Silkscreen-Bold.ttf",
        "sizes": [7, 8, 9, 10, 12, 14],
        "default_size": 8,
        "description": "Pixel font — bold variant",
    },
}

_SYSTEM_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


@lru_cache(maxsize=1)
def _find_system_font() -> str | None:
    for path in _SYSTEM_FONT_PATHS:
        if os.path.isfile(path):
            return path
    return None


def get_available_fonts() -> list[dict]:
    """Return the list of available fonts for the API."""
    fonts: list[dict] = []

    for font_id, info in _BUNDLED_FONTS.items():
        font_path = _FONTS_DIR / info["file"]
        if font_path.is_file():
            fonts.append({
                "id": font_id,
                "name": info["name"],
                "sizes": info["sizes"],
                "default_size": info["default_size"],
                "description": info["description"],
            })

    sys_font = _find_system_font()
    if sys_font:
        name = Path(sys_font).stem.replace("-", " ").replace("_", " ")
        fonts.append({
            "id": "system_sans",
            "name": name,
            "sizes": [8, 10, 12, 14, 16, 20, 24],
            "default_size": 14,
            "description": f"System font: {name}",
        })

    fonts.append({
        "id": "default",
        "name": "Default",
        "sizes": [8, 10, 12, 14, 16, 20, 24],
        "default_size": 14,
        "description": "Pillow built-in font",
    })

    return fonts


@lru_cache(maxsize=32)
def _load_font(font_id: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font by registry ID at the given render size."""
    if font_id in _BUNDLED_FONTS:
        font_path = _FONTS_DIR / _BUNDLED_FONTS[font_id]["file"]
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size)

    if font_id == "system_sans":
        sys_path = _find_system_font()
        if sys_path:
            return ImageFont.truetype(sys_path, size)

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_char(
    char: str,
    resolution: int = 10,
    font_id: str = "press_start_2p",
    font_size: int = 8,
) -> np.ndarray:
    """Render a single character as a binary pixel grid.

    Renders at high resolution using a real TrueType font, then downscales
    with BOX resampling for clean results at low pixel counts.

    Args:
        char: Single character to render.
        resolution: Output grid size (square, e.g. 10 for 10x10).
        font_id: Font registry ID (e.g. "press_start_2p").
        font_size: Font size in target-resolution pixels (e.g. 8 means
                   the character occupies ~8 pixels of the target grid).

    Returns:
        Binary numpy array of shape (resolution, resolution) with 0.0/1.0.
    """
    render_px = resolution * _RENDER_SCALE
    actual_font_size = font_size * _RENDER_SCALE
    font = _load_font(font_id, max(8, actual_font_size))

    img = Image.new("L", (render_px, render_px), 0)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (render_px - tw) // 2 - bbox[0]
    y = (render_px - th) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)

    small = img.resize((resolution, resolution), Image.Resampling.BOX)

    arr = np.array(small, dtype=np.float32) / 255.0
    return (arr > _THRESHOLD).astype(np.float32)


def apply_white_noise(
    frame: np.ndarray,
    noise_prob: float = _DEFAULT_NOISE_PROB,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Flip random pixels with given probability."""
    if rng is None:
        rng = np.random.default_rng()
    result = frame.copy()
    flip_mask = rng.random(frame.shape) < noise_prob
    result[flip_mask] = 1.0 - result[flip_mask]
    return result


def apply_shift_noise(
    frame: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Displace image by 1 pixel in a random direction. Edges fill with 0."""
    if rng is None:
        rng = np.random.default_rng()
    h, w = frame.shape
    direction = rng.integers(0, 4)
    result = np.zeros_like(frame)
    if direction == 0:
        result[: h - 1, :] = frame[1:, :]
    elif direction == 1:
        result[1:, :] = frame[: h - 1, :]
    elif direction == 2:
        result[:, : w - 1] = frame[:, 1:]
    else:
        result[:, 1:] = frame[:, : w - 1]
    return result
