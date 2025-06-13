import cv2
import os
import numpy as np
import json
from .quality_evaluation import evaluate_quality


DEFAULT_ERROR = {
    'sharpness': 0.1,
    'entropy': 0.1,
    'edge_density': 0.1,
    'residual_skew_angle': 0.1,
}

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
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


def save_outputs(original, processed, output_path_tiff, output_path_thumb=None, copied=False):
    """
    Save the processed TIFF image, a reduced JPG thumbnail, and the quality evaluation JSON.
    Always saves both original and processed images in the thumbnail, and always saves the quality file.
    """
    if copied:
        processed = np.zeros_like(original)  # If copied, processed is an empty image
    # Save the processed TIFF
    cv2.imwrite(output_path_tiff, processed)

    # Ensure both images have the same height
    if original.shape[0] != processed.shape[0]:
        height = min(original.shape[0], processed.shape[0])
        original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        processed = cv2.resize(processed, (int(processed.shape[1] * height / processed.shape[0]), height))

    # Ensure both images have the same type
    if original.shape[2] != processed.shape[2]:
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # --- SEPARAZIONE VISIVA ---
    sep_width = 100  # larghezza separatore
    height = original.shape[0]
    separator = 0 * np.ones((height, sep_width, 3), dtype=np.uint8)  # grigio chiaro

    # Concatenate original, separator, processed
    concatenated_image = cv2.hconcat([original, separator, processed])

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

    # --- Calcola e salva la valutazione della qualità ---
    quality = evaluate_quality(original, processed)
    quality_dir = os.path.join(os.path.dirname(output_path_thumb), 'quality')
    quality_dir = os.path.abspath(quality_dir)
    os.makedirs(quality_dir, exist_ok=True)
    quality_filename = os.path.splitext(thumbnail_filename)[0] + '.json'
    quality_path = os.path.join(quality_dir, quality_filename)
    def to_python_type(obj):
        if isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    quality_py = to_python_type(quality)
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(quality_py, f, indent=2, ensure_ascii=False)

    return thumbnail