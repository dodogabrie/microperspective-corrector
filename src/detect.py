import cv2
import numpy as np
from .utils import show_image

def find_page_contour(thresh, show_step_by_step=False, original_image=None):
    """
    Detect the largest contour that resembles a page.

    Parameters:
        thresh (numpy.ndarray): A binary thresholded image where the contours will be detected.
        show_step_by_step (bool): If True, intermediate images will be displayed for debugging purposes.
        original_image (numpy.ndarray): Original colored image for overlay visualization.

    Returns:
        numpy.ndarray: An approximated polygonal contour of the detected page-like shape.

    Raises:
        ValueError: If no page-like contour is found in the image.

    Steps:
        1. Dilate the thresholded image to close small gaps in the contours.
        2. Detect all external contours in the dilated image.
        3. Sort the contours by area in descending order.
        4. Approximate each contour to reduce the number of points.
        5. Check if the approximated contour has at least four vertices, indicating a page-like shape.
        6. If a suitable contour is found, return it. Otherwise, raise an error.
    """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    if show_step_by_step:
        show_image(dilated, "Dilated")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 4:
            if show_step_by_step:
                if original_image is not None:
                    # Create overlay on colored image with high contrast contour
                    overlay = original_image.copy()
                    # Draw thick white border first for contrast
                    cv2.drawContours(overlay, [approx], -1, (255, 255, 255), 15)
                    # Draw thinner colored contour on top
                    cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 8)
                    show_image(overlay, "Detected Contour on Original Image")
                else:
                    # Fallback to binary visualization
                    temp_image = np.zeros_like(thresh)
                    cv2.drawContours(temp_image, [approx], -1, (255, 255, 255), 3)
                    show_image(temp_image, "Detected Contour")
            return approx

    # If no suitable contour is found, return None
    return None