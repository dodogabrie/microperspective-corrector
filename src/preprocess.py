import cv2
from .utils import show_image

def preprocess_image(image, show_step_by_step=False):
    """Convert the image to grayscale, blur it, and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_step_by_step:
        show_image(gray, "Grayscale")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if show_step_by_step:
        show_image(blurred, "Blurred")

    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    if show_step_by_step:
        show_image(thresh, "Thresholded")

    return thresh