import cv2
import os
import numpy as np

def show_image(image, title="Image", max_width=1280, max_height=720, file_path=None):
    """
    Resize and display an image with a title. Waits for a key press to close.

    Args:
        image (numpy.ndarray): Image to display.
        title (str): Window title for the displayed image.
        max_width (int): Maximum width for resizing.
        max_height (int): Maximum height for resizing.
        file_path (str): Path to save the image.
        image_description (str): Description of the image.

    Returns:
        None
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1)
    resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

    if not file_path:
        cv2.imshow(title, resized_image)
        cv2.waitKey(0)

    # this is for educational purposes
    if file_path:
        cv2.imwrite(file_path, resized_image)


def load_image(image_path):
    """Load the input image from the given path."""
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def save_outputs(original, warped, output_path_tiff, output_path_thumb=None):
    """Save the cropped TIFF image and a reduced JPG thumbnail."""
    # Save the TIFF
    # Fallback logic: check if warped is too small, and if so, use original
    final_image, is_fallback = fallback_image(original, warped, return_status=True)
    cv2.imwrite(output_path_tiff, final_image)

    # Ensure both images have the same height
    if original.shape[0] != final_image.shape[0]:
        height = min(original.shape[0], final_image.shape[0])
        original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        final_image = cv2.resize(final_image, (int(final_image.shape[1] * height / final_image.shape[0]), height))

    # Ensure both images have the same type
    if original.shape[2] != final_image.shape[2]:
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(final_image.shape) == 2:
            final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

    # --- SEPARAZIONE VISIVA ---
    # Crea uno sfondo grigio tra le due immagini
    sep_width = 100  # larghezza separatore
    height = original.shape[0]
    separator = 200 * np.ones((height, sep_width, 3), dtype=np.uint8)

    # Concatenate original, separator, final_image (possibly with red background)
    if is_fallback:
        # Add a red background behind the fallback image (right side)
        red_bg = np.zeros_like(final_image)
        red_bg[:, :] = (0, 0, 255)  # BGR for red
        margin = 10
        h, w = final_image.shape[:2]
        overlay = red_bg.copy()
        y1, y2 = margin, h - margin
        x1, x2 = margin, w - margin
        resized_orig = cv2.resize(original, (x2 - x1, y2 - y1))
        overlay[y1:y2, x1:x2] = resized_orig
        concatenated_image = cv2.hconcat([original, separator, overlay])
    else:
        concatenated_image = cv2.hconcat([original, separator, final_image])

    resize_val = 500
    height, width = concatenated_image.shape[:2]
    thumbnail = cv2.resize(concatenated_image, (resize_val, int(resize_val * height / width)))

    # Se non viene passato output_path_thumb, salva in una cartella tmp accanto all'output tiff
    if not output_path_thumb:
        output_dir = os.path.dirname(output_path_tiff)
        output_path_thumb = os.path.join(output_dir, 'tmp')
        os.makedirs(output_path_thumb, exist_ok=True)

    # Extract the base filename and replace the extension
    base_filename = os.path.basename(output_path_tiff)
    thumbnail_filename = base_filename.replace('.tif', '.jpg').replace('.tiff', '.jpg')
    cv2.imwrite(os.path.join(output_path_thumb, thumbnail_filename), thumbnail)

    return thumbnail

def fallback_image(original, final_image, return_status=False):
    """
    Fallback to the original image if the final image is too small.

    Args:
        original (numpy.ndarray): The original image.
        final_image (numpy.ndarray): The processed final image.
        return_status (bool): If True, return (image, is_fallback).

    Returns:
        numpy.ndarray or (numpy.ndarray, bool): The image to use, and optionally the fallback status.
    """
    # Calculate the area of the original and final images
    original_area = original.shape[0] * original.shape[1]
    final_area = final_image.shape[0] * final_image.shape[1]
    is_fallback = final_area <= 0.1 * original_area
    if return_status:
        return (original if is_fallback else final_image), is_fallback
    else:
        return original if is_fallback else final_image