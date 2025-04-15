"""
ECG Image Processing Application (Simplified Interface)

This application provides a complete pipeline for processing ECG images:
1. Remove grid lines from ECG images
2. Remove text and frame elements
3. Segment the image into individual ECG leads
4. Extract and visualize the ECG signal

Usage:
    python simplified_main.py <input_image> <output_directory>
"""

import sys
from ecg_processor import ECGProcessor


def main():
    """Entry point for ECG processing application."""
    if len(sys.argv) < 3:
        print("Usage: python simplified_main.py <input_image> <output_directory>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        # Create processor with debug enabled
        processor = ECGProcessor(debug=True)

        # Process the ECG image
        processor.process(input_image_path, output_dir)

        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error processing ECG image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()