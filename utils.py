"""Utilities for ECG image processing."""

import os
import cv2
import numpy as np


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_debug_image(image, output_dir, filename):
    """Save image to output directory for debugging."""
    if output_dir:
        ensure_directory_exists(output_dir)
        cv2.imwrite(os.path.join(output_dir, filename), image)


def to_grayscale(image):
    """Convert image to grayscale if it's not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def ensure_white_signal_on_black_background(image):
    """Ensure image has white signal on black background."""
    if np.mean(image) > 127:
        return 255 - image
    return image.copy()


def binarize_image(image, threshold=127):
    """Convert grayscale image to binary using threshold."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary