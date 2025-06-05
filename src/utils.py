import cv2
import os

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
    cv2.imwrite(output_path_tiff, warped)

    # Ensure both images have the same height
    if original.shape[0] != warped.shape[0]:
        height = min(original.shape[0], warped.shape[0])
        original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        warped = cv2.resize(warped, (int(warped.shape[1] * height / warped.shape[0]), height))

    # Ensure both images have the same type
    if original.shape[2] != warped.shape[2]:
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(warped.shape) == 2:
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # Concatenate original and warped images
    concatenated_image = cv2.hconcat([original, warped])

    # Resize the concatenated image to create a thumbnail
    resize_val = 500
    height, width = concatenated_image.shape[:2]
    thumbnail = cv2.resize(concatenated_image, (resize_val, int(resize_val * height / width)))

    if not output_path_thumb:
        output_path_thumb = 'tmp'
        os.makedirs(output_path_thumb, exist_ok=True)

    # Extract the base filename and replace the extension
    base_filename = os.path.basename(output_path_tiff)
    thumbnail_filename = base_filename.replace('.tif', '.jpg').replace('.tiff', '.jpg')

    cv2.imwrite(os.path.join(output_path_thumb, thumbnail_filename), thumbnail)

    return thumbnail


def fallback_image(original, final_image):
    """
    Fallback to the original image if the final image is too small.

    Args:
        original (numpy.ndarray): The original image.
        final_image (numpy.ndarray): The processed final image.

    Returns:
        numpy.ndarray: The original image if the final image is too small, otherwise the final image.
    """
    # Calculate the area of the original and final images
    original_area = original.shape[0] * original.shape[1]
    final_area = final_image.shape[0] * final_image.shape[1]

    # Check if the final image is less than or equal to half the size of the original
    if final_area <= 0.5 * original_area:
        return original
    else:
        return final_image