import cv2
from .utils import show_image

def preprocess_image(image, show_step_by_step=False):
    """Convert the image to grayscale, blur it, and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_step_by_step:
        show_image(gray, "Grayscale")

    # Determine adaptive kernel size based on image size
    min_dim = min(gray.shape[:2])
    # Kernel size is 1/50th of min dimension, rounded to nearest odd integer >= 3
    k = max(3, int((min_dim / 50) // 2 * 2 + 1))
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    if show_step_by_step:
        show_image(blurred, f"Blurred (kernel={k}x{k})")

    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    if show_step_by_step:
        show_image(thresh, "Thresholded")

    return thresh