import cv2
import numpy as np
from .utils import show_image
from .crop import remove_lateral_blacks


def warp_image(image, page_contour, border_pixels=0, show_step_by_step=False):
    """
    Warp the image to make the detected page rectangular, with an optional external border.

    Args:
        image (numpy.ndarray): Original image.
        page_contour (numpy.ndarray): Detected page contour.
        border_pixels (float): Percentage to expand the box.
        show_step_by_step (bool): If True, displays the warped image.

    Returns:
        numpy.ndarray: Warped image with the optional border.
    """

    copy_image = image.copy()
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(page_contour)
    center_box = rect[0]
    angle = rect[2]

    if angle > 80:
        angle = angle - 90


    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Calcola la matrice di rotazione attorno al centro del box
    M = cv2.getRotationMatrix2D(center_box, angle, 1.0)
    rotated = cv2.warpAffine(copy_image, M, (image.shape[1], image.shape[0]))
    rotated_white = cv2.warpAffine(copy_image, M, (image.shape[1], image.shape[0]),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255)
                                    )

    # Trasforma i punti del box con la matrice di rotazione
    rotated_box = cv2.transform(np.array([box], dtype="float32"), M)[0]
    x, y, w, h = cv2.boundingRect(rotated_box)
    # Expand the box by border_pixels variable. Avoid going under zero
    x = max(0, int(x - border_pixels))
    y = max(0, int(y - border_pixels))
    h = h + int(border_pixels * 2)
    w = w + int(border_pixels * 2)
    cropped = rotated[y:y+h, x:x+w]
    cropped_white = rotated_white[y:y+h, x:x+w]

    if show_step_by_step:
        show_image(cropped, "Immagine Ruotata e Croppata")

    warped = remove_lateral_blacks(cropped, cropped_white, show_step_by_step, False)

    return warped