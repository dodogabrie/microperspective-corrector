import numpy as np
import os
from src.utils import show_image, load_image, save_outputs
from src.preprocess import preprocess_image
from src.detect import find_page_contour
from src.transform import warp_image
# from src.crop import remove_lateral_blacks
# from src.quality_evaluation import evaluate_quality

# Input and output directories
DATA_DIR = 'dataset/original'
OUTPUT_DIR = 'dataset/output'
OUTPUT_THUMB_DIR = 'dataset/output_thumb'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_THUMB_DIR, exist_ok=True)


def process_tiff(image_path, output_path_tiff, output_path_thumb, border_pixels=0, show_step_by_step=False,
                 show_before_after=False):
    """
    Full pipeline to process a TIFF image with metadata preservation.

    This function processes a TIFF image by loading it, detecting the page contour,
    warping the image to correct perspective, and optionally displaying intermediate steps.
    The processed image is saved to the specified output paths.

    Args:
        image_path (str): Path to the input TIFF image.
        output_path_tiff (str): Path to save the cropped TIFF image.
        output_path_thumb (str): Path to save the reduced thumbnail.
        border_pixels (int): Number of pixels for the external border.
        show_step_by_step (bool): If True, shows intermediate steps of the processing.
        show_before_after (bool): If True, shows the original and processed images.

    Returns:
        None

    Raises:
        ValueError: If no page-like contour is found in the image.

    Notes:
        - The function uses several utility functions to handle image loading, processing, and saving.
        - The processed image is warped to correct perspective distortions.
    """
    print(f"Processing image: {image_path}")
    image = load_image(image_path)
    if show_before_after:
        show_image(image, "Original Image")
    thresh, border_value = preprocess_image(image, show_step_by_step)
    page_contour = find_page_contour(thresh, show_step_by_step)
    copied = False
    if page_contour is None:
        copied = True
        warped = np.copy(image)
        print("No page-like contour found, returning empty image.")
        thumbnail = save_outputs(image, warped, output_path_tiff, output_path_thumb, 
                                copied=copied, original_path=image_path) 
    else:
        warped, no_cropped = warp_image(image, page_contour, border_pixels, show_step_by_step, border_value=border_value)
        print("Found countour, saving cropped/rotated image.")
        thumbnail = save_outputs(image, warped, output_path_tiff, output_path_thumb, 
                                copied=copied, output_no_cropped=no_cropped, original_path=image_path)
    if show_before_after:
        show_image(warped, "Cropped Image")
    
    return thumbnail
