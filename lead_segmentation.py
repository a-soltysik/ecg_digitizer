"""ECG lead segmentation functionality."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils import ensure_white_signal_on_black_background, binarize_image


def find_signal_points(binary_image):
    """Identify all signal points across the image."""
    height, width = binary_image.shape
    all_signal_points = []

    for col in range(width):
        # Find all white pixels in this column
        white_pixels = np.where(binary_image[:, col] > 0)[0]

        # Group adjacent pixels to find signal points
        if len(white_pixels) > 0:
            groups = []
            current_group = [white_pixels[0]]

            for i in range(1, len(white_pixels)):
                if white_pixels[i] - white_pixels[i - 1] <= 1:
                    current_group.append(white_pixels[i])
                else:
                    signal_y = int(np.median(current_group))
                    all_signal_points.append(signal_y)
                    current_group = [white_pixels[i]]

            # Add the last group
            if current_group:
                signal_y = int(np.median(current_group))
                all_signal_points.append(signal_y)

    return all_signal_points


def cluster_lead_positions(signal_points, eps=10):
    """Cluster signal points to identify lead positions."""
    X = np.array(signal_points).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=3).fit(X)

    # Extract cluster centers (lead positions)
    lead_positions = []
    for label in set(clustering.labels_):
        if label != -1:  # Skip noise points
            cluster_points = X[clustering.labels_ == label]
            lead_positions.append(np.median(cluster_points))

    return sorted(lead_positions)


def calculate_lead_boundaries(lead_positions, image_height):
    """Calculate boundaries between leads based on midpoints."""
    lead_boundaries = [0]  # Start with top of image

    for i in range(len(lead_positions) - 1):
        boundary = int((lead_positions[i] + lead_positions[i + 1]) / 2)
        lead_boundaries.append(boundary)

    lead_boundaries.append(image_height)  # Add bottom of image
    return lead_boundaries


def adjust_to_expected_lead_count(lead_boundaries, expected_leads=12):
    """Adjust lead boundaries to match expected number of leads."""
    if len(lead_boundaries) - 1 != expected_leads:
        print(f"Warning: Found {len(lead_boundaries) - 1} leads, expected {expected_leads}")

        # If too many leads, discard the last ones instead of merging
        if len(lead_boundaries) - 1 > expected_leads:
            print(f"Discarding {len(lead_boundaries) - 1 - expected_leads} excess leads")
            # Keep only the first expected_leads+1 boundaries (for expected_leads leads)
            lead_boundaries = lead_boundaries[:expected_leads + 1]

    return lead_boundaries


def visualize_segmentation(binary_image, lead_positions, lead_boundaries):
    """Visualize segmentation results."""
    plt.figure(figsize=(12, 10))
    plt.imshow(binary_image, cmap='gray')

    # Plot detected lead positions
    for pos in lead_positions:
        plt.axhline(pos, color='blue', linestyle='-', alpha=0.5)

    # Plot lead boundaries
    for boundary in lead_boundaries:
        plt.axhline(boundary, color='red', linestyle='--', alpha=0.7)

    plt.title(
        f'Detected {len(lead_positions)} Lead Positions (blue) and {len(lead_boundaries) - 1} Lead Segments (red)')
    plt.savefig('lead_segmentation.png')
    plt.show()



def segment_leads(image, debug=True, eps=10, expected_leads=12):
    """Segment ECG image into leads."""
    # Ensure consistent image format
    image_processed = ensure_white_signal_on_black_background(image)
    binary_image = binarize_image(image_processed)
    height, width = binary_image.shape

    # Find signal points
    signal_points = find_signal_points(binary_image)

    # Cluster to find lead positions
    lead_positions = cluster_lead_positions(signal_points, eps)

    # Calculate lead boundaries
    lead_boundaries = calculate_lead_boundaries(lead_positions, height)

    # Adjust to expected number of leads
    lead_boundaries = adjust_to_expected_lead_count(lead_boundaries, expected_leads)

    # Visualize if in debug mode
    if debug:
        visualize_segmentation(binary_image, lead_positions, lead_boundaries)

    return lead_boundaries, lead_positions, binary_image