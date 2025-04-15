"""
ECG Image Processing Application

Usage:
    python main.py <input_image> <output_directory>
"""

import sys
from ecg_processor import ECGProcessor


def main():
    """Entry point for ECG processing application."""
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_image> <output_directory>")
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