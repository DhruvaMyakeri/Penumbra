"""Generate reference-image assets for X2's `set_reference_image`.

X2 takes a reference image of "a character or object to insert / swap in". These are
drawn rather than photographed so the asset is reproducible from source and carries no
licensing question, and so the intended object is unambiguous: a solid opaque thing
with clean silhouette edges is what an occlusion intervention needs.

Each asset is a product-shot style image - object centred on a plain light background,
which is the shape conditioning models expect.

    python tools/make_occluder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from penumbra.config import REPO_ROOT  # noqa: E402

ASSETS = REPO_ROOT / "assets"
SIZE = 512


def _canvas() -> np.ndarray:
    img = np.full((SIZE, SIZE, 3), 240, np.uint8)
    # A soft vignette reads as a studio backdrop rather than a flat fill.
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    r = np.sqrt(((yy - SIZE / 2) / SIZE) ** 2 + ((xx - SIZE / 2) / SIZE) ** 2)
    img = np.clip(img.astype(np.float32) - (r * 60)[..., None], 180, 255).astype(np.uint8)
    return img


def cardboard_box() -> np.ndarray:
    """A closed cardboard carton, three-quarter view. Opaque, hard edges, familiar."""
    img = _canvas()
    front = np.array([[130, 200], [130, 400], [330, 430], [330, 230]], np.int32)
    top = np.array([[130, 200], [330, 230], [400, 170], [200, 145]], np.int32)
    side = np.array([[330, 230], [400, 170], [400, 370], [330, 430]], np.int32)
    cv2.fillPoly(img, [front], (168, 132, 88))
    cv2.fillPoly(img, [top], (205, 170, 120))
    cv2.fillPoly(img, [side], (132, 100, 64))
    for poly in (front, top, side):
        cv2.polylines(img, [poly], True, (92, 68, 42), 3, cv2.LINE_AA)
    # Tape seam down the front face.
    cv2.line(img, (230, 215), (230, 415), (198, 186, 160), 12, cv2.LINE_AA)
    return img


def blue_toolbox() -> np.ndarray:
    """A saturated blue toolbox - high contrast against a wooden table, and a hue
    the scene does not already contain, which makes it easy to segment when checking
    that the insertion actually landed."""
    img = _canvas()
    body = np.array([[120, 240], [120, 400], [390, 400], [390, 240]], np.int32)
    lid = np.array([[110, 240], [140, 195], [370, 195], [400, 240]], np.int32)
    cv2.fillPoly(img, [body], (40, 40, 190))
    cv2.fillPoly(img, [lid], (30, 30, 150))
    cv2.polylines(img, [body], True, (20, 20, 90), 3, cv2.LINE_AA)
    cv2.polylines(img, [lid], True, (20, 20, 90), 3, cv2.LINE_AA)
    cv2.rectangle(img, (225, 165), (285, 200), (60, 60, 60), -1)
    cv2.rectangle(img, (225, 165), (285, 200), (25, 25, 25), 3, cv2.LINE_AA)
    return img


def steel_can() -> np.ndarray:
    """A plain metal can - a compact occluder that sits naturally on a table."""
    img = _canvas()
    cv2.ellipse(img, (256, 380), (95, 30), 0, 0, 360, (150, 150, 155), -1)
    cv2.rectangle(img, (161, 175), (351, 380), (172, 172, 178), -1)
    cv2.ellipse(img, (256, 175), (95, 30), 0, 0, 360, (198, 198, 205), -1)
    for x, shade in ((185, 205), (300, 140)):
        cv2.line(img, (x, 180), (x, 378), (shade, shade, shade + 4), 16, cv2.LINE_AA)
    cv2.ellipse(img, (256, 175), (95, 30), 0, 0, 360, (110, 110, 118), 3, cv2.LINE_AA)
    cv2.rectangle(img, (161, 250), (351, 300), (60, 90, 170), -1)
    return img


ASSET_BUILDERS = {
    "cardboard_box": cardboard_box,
    "blue_toolbox": blue_toolbox,
    "steel_can": steel_can,
}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, build in ASSET_BUILDERS.items():
        path = ASSETS / f"{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(build(), cv2.COLOR_RGB2BGR))
        print(f"{name:<16} -> {path}  ({path.stat().st_size / 1024:.0f} KB)")
    strip = np.concatenate([build() for build in ASSET_BUILDERS.values()], axis=1)
    contact = ASSETS / "_contact_sheet.png"
    cv2.imwrite(str(contact), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"contact sheet -> {contact}")


if __name__ == "__main__":
    main()
