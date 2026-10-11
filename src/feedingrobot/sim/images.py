"""Shared RGB area downsampling, cropping and fixed range conversion."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _area_weights(source, target):
    boundaries = np.linspace(0., source, target+1)
    pixels = np.arange(source)
    overlap = np.maximum(0., np.minimum(boundaries[1:, None], pixels+1)
                         - np.maximum(boundaries[:-1, None], pixels))
    return overlap / (source/target)


def resize_area(rgb, size=240):
    """Integrate source pixel areas; preserve uint8 storage at the 240 layer."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected HWC uint8 RGB")
    height, width = rgb.shape[:2]
    vertical = _area_weights(height, size) @ rgb.reshape(height, width*3)
    result = np.einsum("ywc,xw->yxc", vertical.reshape(size, width, 3),
                       _area_weights(width, size), optimize=True)
    return np.rint(result).astype(np.uint8)


def rgb240_to_observation(rgb240):
    """One frame, CHW float32 [0,1]; no image normalization is applied here."""
    rgb240 = np.asarray(rgb240)
    if rgb240.shape != (240, 240, 3) or rgb240.dtype != np.uint8:
        raise ValueError("Expected 240x240 HWC uint8 RGB")
    return np.ascontiguousarray(rgb240.transpose(2, 0, 1), dtype=np.float32) / np.float32(255.)


def crop_image(image, size=216, *, training=False, rng=None, offset=None):
    """Crop HWC image; deterministic centre by default, supplied RNG in training."""
    height, width = image.shape[:2]
    if size > min(height, width):
        raise ValueError("Crop is larger than image")
    if offset is None:
        if training:
            if rng is None:
                raise ValueError("Training crop requires an explicit random generator")
            offset = (rng.integers(height-size+1), rng.integers(width-size+1))
        else:
            offset = ((height-size)//2, (width-size)//2)
    top, left = offset
    if not (0 <= top <= height-size and 0 <= left <= width-size):
        raise ValueError("Crop offset outside image")
    return np.ascontiguousarray(image[top:top+size, left:left+size])


def image_to_cnn(rgb240, *, training=False, rng=None):
    """Convert raw uint8 RGB once, crop, then map [0,1] to [-1,1]."""
    observed = rgb240_to_observation(rgb240)
    cropped = crop_image(observed.transpose(1, 2, 0), training=training, rng=rng)
    return np.ascontiguousarray(cropped.transpose(2, 0, 1)) * np.float32(2.) - np.float32(1.)
