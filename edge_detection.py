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
                 show_before_after=False,):
    """
    Full pipeline to process a TIFF image.

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
    thresh = preprocess_image(image, show_step_by_step)
    page_contour = find_page_contour(thresh, show_step_by_step)
    copied = False
    if page_contour is None:
        copied = True
        warped = np.copy(image)
        print("No page-like contour found, returning empty image.")
    else:
        warped, _ = warp_image(image, page_contour, border_pixels, show_step_by_step)
    if show_before_after:
        show_image(warped, "Cropped Image")
    # Call save_outputs with original and warped only; fallback logic is now inside save_outputs
    thumbnail = save_outputs(image, warped, output_path_tiff, output_path_thumb, copied=copied)
    return thumbnail


# def run_all():
#     """
#     Process all TIFF files in the input directory.

#     This function iterates over all TIFF files in the DATA_DIR, processes each file
#     using the `process_tiff` function, and saves the processed images to the OUTPUT_DIR.
#     """
#     for filename in os.listdir(DATA_DIR):
#         if filename.lower().endswith('.tiff') or filename.lower().endswith('.tif'):
#             input_path = os.path.join(DATA_DIR, filename)
#             output_tiff_path = os.path.join(OUTPUT_DIR, filename)
#             output_thumb_path = os.path.join(OUTPUT_THUMB_DIR, f"{os.path.splitext(filename)[0]}.jpg")

#             # Process the selected file
#             process_tiff(input_path, output_tiff_path, output_thumb_path, border_pixels=2, show_step_by_step=False,
#                          show_before_after=True)
#     cv2.destroyAllWindows()


# def run_test():
#     """
#     Process a random TIFF file from the input directory for testing.

#     This function selects a random TIFF file from the DATA_DIR, processes it using
#     the `process_tiff` function, and saves the processed image to the OUTPUT_DIR.
#     """
#     tiff_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]

#     if tiff_files:
#         filename = random.choice(tiff_files)
#         input_path = os.path.join(DATA_DIR, filename)
#         output_tiff_path = os.path.join(OUTPUT_DIR, filename)
#         output_thumb_path = os.path.join(OUTPUT_THUMB_DIR, f"{os.path.splitext(filename)[0]}.jpg")

#         # Process the selected file
#         process_tiff(input_path, output_tiff_path, output_thumb_path, border_pixels=100, show_step_by_step=True,
#                      show_before_after=True)
#     else:
#         print("No TIFF files found in the directory.")

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     run_test()
