"""
ECG Image Processor

This module provides a simplified interface to the ECG image processing pipeline.
It combines the functionality of grid removal, text removal, frame removal,
lead segmentation, and signal extraction.
"""

import os
import cv2
import matplotlib.pyplot as plt

from utils import ensure_directory_exists, to_grayscale, save_debug_image
from grid_removal import remove_grid, remove_isolated_pixels
from text_removal import create_text_mask, remove_text
from frame_removal import remove_frame
from lead_segmentation import segment_leads
from signal_extraction import create_plot


class ECGProcessor:
    """Process ECG images through a complete pipeline."""

    def __init__(self, debug=True):
        """
        Initialize ECG processor.

        Args:
            debug: Whether to save debug images and print extra information
        """
        self.debug = debug

    def process(self, image_path, output_dir):
        """
        Process an ECG image through the full pipeline.

        Args:
            image_path: Path to input ECG image
            output_dir: Directory to save output and debug images

        Returns:
            The processed ECG visualization figure
        """
        # Setup directories
        ensure_directory_exists(output_dir)
        debug_dir = os.path.join(output_dir, "debug") if self.debug else None

        # Load and preprocess image
        gray_image = self._load_and_convert_image(image_path, debug_dir)

        # Process grid, text, and frame
        cleaned_image = self._process_image_elements(gray_image, debug_dir)

        # Segment leads and visualize results
        figure = self._segment_and_visualize(cleaned_image, debug_dir, output_dir)

        print(f"ECG processing complete. Results saved in '{output_dir}'")
        return figure

    def _load_and_convert_image(self, image_path, debug_dir):
        """Load image from path and convert to grayscale."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from {image_path}")

        gray_image = to_grayscale(image)
        save_debug_image(gray_image, debug_dir, "01_original.png")
        return gray_image

    def _process_image_elements(self, gray_image, debug_dir):
        """Remove grid, text, and frame from the image."""
        # Remove grid
        grid_removed = remove_grid(gray_image, debug_dir)
        save_debug_image(grid_removed, debug_dir, "02_grid_removed.png")

        # Remove text
        text_mask = create_text_mask(grid_removed, debug_dir)
        text_removed = remove_text(grid_removed, text_mask)
        save_debug_image(text_removed, debug_dir, "03_text_removed.png")

        # Remove frame
        frame_removed = remove_frame(text_removed, debug_dir)
        save_debug_image(frame_removed, debug_dir, "04_frame_removed.png")

        # Clean isolated pixels
        cleaned = remove_isolated_pixels(frame_removed)
        save_debug_image(cleaned, debug_dir, "05_cleaned.png")

        return cleaned

    def _segment_and_visualize(self, cleaned_image, debug_dir, output_dir):
        """Segment the image into leads and create visualization."""
        # Segment leads
        lead_boundaries, lead_positions, binary_image = segment_leads(
            cleaned_image, debug=self.debug
        )

        # Create visualization
        ecg_figure = create_plot(
            binary_image, lead_boundaries, lead_positions, debug=self.debug
        )

        # Save figure
        ecg_figure.savefig(os.path.join(output_dir, "digitized_ecg.png"), dpi=300)

        return ecg_figure