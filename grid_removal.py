"""Grid line removal from ECG images."""

import cv2
import numpy as np
from utils import save_debug_image

def apply_morphological_filtering(image, output_dir=None):
    """Apply morphological operations to remove grid."""
    kernel_open = np.ones((3, 3), np.uint8)
    morph_open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)

    # Threshold using Otsu's method
    _, morph_thresh = cv2.threshold(
        morph_open, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    save_debug_image(morph_thresh, output_dir, "morph_filtered.png")
    return morph_thresh


def remove_isolated_pixels(image, connectivity=8, min_connections=1):
    """Remove isolated pixels while preserving trace lines."""
    cleaned = image.copy()
    height, width = image.shape

    # Define neighbor offsets based on connectivity
    if connectivity == 8:
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
    else:  # 4-way connectivity
        neighbor_offsets = [
            (-1, 0), (0, -1), (0, 1), (1, 0)
        ]

    # Find white pixels
    white_pixels = np.where(image == 255)
    coordinates = list(zip(white_pixels[0], white_pixels[1]))

    # Check each white pixel
    for y, x in coordinates:
        white_neighbors = 0

        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx

            if 0 <= ny < height and 0 <= nx < width and image[ny, nx] == 255:
                white_neighbors += 1

        # Remove pixels with fewer connections than required
        if white_neighbors < min_connections:
            cleaned[y, x] = 0

    return cleaned


def remove_grid(image, output_dir=None):
    """Apply multiple methods to remove grid from ECG image."""
    save_debug_image(image, output_dir, "original_gray.png")
    return apply_morphological_filtering(image, output_dir)