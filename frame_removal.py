"""Frame detection and removal from ECG images."""

import cv2
import numpy as np
from utils import save_debug_image


def detect_edges(image):
    """Detect edges for frame detection."""
    edges = cv2.Canny(image, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def find_largest_contour(edge_image):
    """Find the largest contour in an edge image."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort contours by area in descending order
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]


def get_frame_rectangle(contour, min_area=100):
    """Get a simplified rectangle from a contour representing a frame."""
    if contour is None or cv2.contourArea(contour) <= min_area:
        return None

    # Approximate the contour to get a simpler polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Get the bounding rectangle
    return cv2.boundingRect(approx)


def crop_to_frame_interior(image, rect, margin=10):
    """Crop image to the interior of a detected frame rectangle."""
    if rect is None:
        return image

    x, y, w, h = rect
    img_h, img_w = image.shape[:2]

    # Apply a margin to ensure we're inside the frame
    x += margin
    y += margin
    w -= 2 * margin
    h -= 2 * margin

    # Ensure valid coordinates
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    # Check dimensions
    if w <= 0 or h <= 0:
        return image

    return image[y:y + h, x:x + w]


def apply_margin_crop(image, margin_percent=0.05):
    """Apply a simple margin crop as fallback when frame detection fails."""
    h, w = image.shape[:2]

    margin_x = min(int(w * margin_percent), w // 10)
    margin_y = min(int(h * margin_percent), h // 10)

    # Calculate valid crop dimensions
    crop_width = w - 2 * margin_x
    crop_height = h - 2 * margin_y

    # Ensure dimensions are positive
    if crop_width <= 0 or crop_height <= 0:
        return image

    return image[margin_y:h - margin_y, margin_x:w - margin_x]


def remove_frame(image, output_dir=None):
    """Detect and remove rectangular frame from an ECG image."""
    try:
        # Detect edges for frame finding
        dilated_edges = detect_edges(image)
        save_debug_image(dilated_edges, output_dir, "dilated_edges.png")

        # Find and process the largest contour
        largest_contour = find_largest_contour(dilated_edges)
        rect = get_frame_rectangle(largest_contour)

        if rect is not None:
            # Get dimensions
            x, y, w, h = rect
            # If the rectangle seems reasonable
            if 20 < w < image.shape[1] and 20 < h < image.shape[0]:
                cropped = crop_to_frame_interior(image, rect)
                save_debug_image(cropped, output_dir, "cropped.png")
                return cropped
    except Exception as e:
        print(f"Warning: Frame removal failed: {str(e)}")

    # Fallback to simple margin crop
    print("Using fallback margin-based cropping.")
    cropped = apply_margin_crop(image)
    save_debug_image(cropped, output_dir, "fallback_cropped.png")

    return cropped