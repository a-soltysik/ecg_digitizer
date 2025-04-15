"""Text detection and removal from ECG images."""

import cv2
import numpy as np
from utils import save_debug_image


def create_margin_mask(image, left_margin_percent=0.1, bottom_margin_percent=0.05):
    """Create a mask for margins where text typically appears in ECG images."""
    height, width = image.shape
    mask = np.zeros_like(image)

    # Left margin for lead labels
    left_margin_width = int(width * left_margin_percent)
    mask[:, 0:left_margin_width] = 255

    # Bottom margin for scale and info
    bottom_margin_height = int(height * bottom_margin_percent)
    mask[height - bottom_margin_height:height, :] = 255

    return mask


def find_text_by_connected_components(image):
    """Find text elements using connected component analysis."""
    height, width = image.shape
    text_mask = np.zeros_like(image)

    # Invert the image (text is typically white on dark background)
    inverted = cv2.bitwise_not(image)

    # Apply threshold to separate text from background
    _, thresh = cv2.threshold(inverted, 170, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=8, ltype=cv2.CV_32S
    )

    if num_labels <= 1:  # No components found
        return text_mask

    # Analyze component sizes
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
    median_area = np.median(areas)
    small_threshold = median_area * 0.1

    # Define size thresholds for text
    max_text_width = width // 12
    max_text_height = height // 15

    # Mark components likely to be text
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width_component = stats[i, cv2.CC_STAT_WIDTH]
        height_component = stats[i, cv2.CC_STAT_HEIGHT]

        if (area < small_threshold and
                width_component < max_text_width and
                height_component < max_text_height):
            text_mask[labels == i] = 255

    return text_mask


def create_text_mask(image, output_dir=None):
    """Create a comprehensive mask for all text elements in the image."""
    # Get position-based mask (margins)
    text_mask = create_margin_mask(image)
    save_debug_image(text_mask, output_dir, "position_mask.png")

    try:
        # Get text detected by component analysis
        cc_text_mask = find_text_by_connected_components(image)
        save_debug_image(cc_text_mask, output_dir, "cc_text_mask.png")

        # Combine masks
        text_mask = cv2.bitwise_or(text_mask, cc_text_mask)
    except Exception as e:
        print(f"Warning: Connected component analysis failed: {str(e)}")

    # Dilate to ensure complete coverage
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    save_debug_image(text_mask, output_dir, "final_text_mask.png")
    return text_mask


def remove_text(image, text_mask):
    """Remove text from the image using the provided text mask."""
    cleaned = image.copy()
    cleaned[text_mask > 0] = 0  # Replace text with black
    return cleaned