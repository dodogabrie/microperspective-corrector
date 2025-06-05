import cv2
import numpy as np
from .utils import show_image

def find_page_contour(thresh, show_step_by_step=False):
    """
    Detect the largest rectangular box that resembles a page.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    if show_step_by_step:
        show_image(dilated, "Dilated")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_box = None

    img_area = thresh.shape[0] * thresh.shape[1]
    
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    
        area = cv2.contourArea(box)
    
        # SCARTA box troppo grande (praticamente tutta l'immagine)
        if area > 0.98 * img_area:
            continue
    
        if area > max_area:
            max_area = area
            best_box = box

    if best_box is None:
        print("No suitable page-like contour found")

    if show_step_by_step:
    
        temp_image = cv2.cvtColor(dilated.copy(), cv2.COLOR_GRAY2BGR)
    
        # Resize per visualizzazione (non modifico best_box originale)
        max_dim = 1200
        h, w = temp_image.shape[:2]
        scale = max_dim / max(h, w)
        resized_image = cv2.resize(temp_image, (int(w * scale), int(h * scale)))
    
        # Scala anche il box per disegnare
        scaled_box = (best_box * scale).astype(int)
    
        thickness = max(3, int(min(resized_image.shape[0], resized_image.shape[1]) / 80))
        cv2.drawContours(resized_image, [scaled_box], -1, (0, 255, 255), thickness)
    
        show_image(resized_image, f"Detected Page Box (area={max_area})")


    return best_box
