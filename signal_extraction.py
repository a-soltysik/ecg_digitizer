"""Extract ECG signal from segmented leads."""

import numpy as np
import matplotlib.pyplot as plt


def extract_lead_boundaries(binary_image, y_start, y_end, baseline_y=None):
    """
    Extract the boundaries of a lead signal.

    For each column, it extracts:
    - Topmost pixel (minimum y) for upward peaks
    - Bottommost pixel (maximum y) for downward troughs
    """
    height, width = binary_image.shape

    # Extract lead region
    lead_img = binary_image[y_start:y_end, :]
    lead_height = y_end - y_start

    # Use provided baseline or default to middle of segment
    if baseline_y is None:
        baseline_y = lead_height // 2
    else:
        baseline_y = baseline_y - y_start  # Make relative to segment

    # Initialize arrays for boundary points
    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []

    # Process each column
    for x in range(width):
        # Find white pixels in this column
        white_pixels = np.where(lead_img[:, x] > 0)[0]

        if len(white_pixels) > 0:
            if len(white_pixels) > 1:
                # Keep topmost and bottommost pixels
                top_y = np.min(white_pixels)
                bottom_y = np.max(white_pixels)

                # Add to upper boundary (convert to amplitude)
                x_upper.append(x)
                y_upper.append(baseline_y - top_y)

                # Add to lower boundary
                x_lower.append(x)
                y_lower.append(baseline_y - bottom_y)
            else:
                # Single pixel case
                y = white_pixels[0]
                amplitude = baseline_y - y

                if amplitude >= 0:  # Above baseline
                    x_upper.append(x)
                    y_upper.append(amplitude)
                else:  # Below baseline
                    x_lower.append(x)
                    y_lower.append(amplitude)

    # Convert to arrays and sort by x-coordinate
    upper_boundary = sort_boundary_points(x_upper, y_upper)
    lower_boundary = sort_boundary_points(x_lower, y_lower)

    return upper_boundary, lower_boundary, lead_height


def sort_boundary_points(x_points, y_points):
    """Sort boundary points by x-coordinate."""
    if not x_points:
        return np.array([]), np.array([])

    # Convert to numpy arrays
    x_array = np.array(x_points)
    y_array = np.array(y_points)

    # Sort by x-coordinate
    sort_idx = np.argsort(x_array)
    x_sorted = x_array[sort_idx]
    y_sorted = y_array[sort_idx]

    return x_sorted, y_sorted


def plot_lead(ax, upper_boundary, lower_boundary, lead_height, lead_label, debug=False):
    """Plot a single lead on the provided axis."""
    x_upper, y_upper = upper_boundary
    x_lower, y_lower = lower_boundary

    # Plot boundaries
    if len(x_upper) > 0:
        ax.plot(x_upper, y_upper, 'k-', linewidth=1)

    if len(x_lower) > 0:
        ax.plot(x_lower, y_lower, 'k-', linewidth=1)

    # Set title
    ax.set_title(lead_label, fontweight='bold')

    # Set y-axis range
    max_amplitude = lead_height // 2
    ax.set_ylim([-max_amplitude * 0.8, max_amplitude * 1.2])

    # Add baseline
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Remove y ticks
    ax.set_yticks([])

    # Add debug info if requested
    if debug:
        upper_count = len(x_upper)
        lower_count = len(x_lower)
        ax.text(len(x_upper) * 0.02, max_amplitude * 0.8,
                f"Height: {lead_height}px | Upper: {upper_count} | Lower: {lower_count}",
                bbox=dict(facecolor='white', alpha=0.7))


def create_plot(binary_image, lead_boundaries, lead_positions, debug=False):
    """Create plot with all ECG leads."""
    lead_count = len(lead_boundaries) - 1

    # Create figure with subplots
    fig, axes = plt.subplots(lead_count, 1, figsize=(15, 2 * lead_count), sharex=True)
    if lead_count == 1:
        axes = [axes]  # Make axes iterable for single lead

    # Lead names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Process each lead
    for i in range(lead_count):
        # Get lead boundaries
        y_start = lead_boundaries[i]
        y_end = lead_boundaries[i + 1]

        # Get baseline (lead position)
        baseline_y = lead_positions[i] if i < len(lead_positions) else None

        # Extract boundary points
        upper_boundary, lower_boundary, lead_height = extract_lead_boundaries(
            binary_image, y_start, y_end, baseline_y
        )

        # Get lead label
        lead_label = lead_names[i] if i < len(lead_names) else f'Lead {i + 1}'

        # Plot this lead
        plot_lead(axes[i], upper_boundary, lower_boundary, lead_height, lead_label, debug)

    # Add x-axis label to bottom plot
    axes[-1].set_xlabel('Sample (pixel column)')

    # Add overall title
    plt.suptitle('Digitized 12-Lead ECG (Boundary Pixels Only)', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Print debug info
    if debug:
        print(f"Total leads plotted: {lead_count}")
        print(f"Image dimensions: {binary_image.shape[1]}x{binary_image.shape[0]} pixels")

    return fig