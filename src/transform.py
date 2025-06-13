import pyvips
import numpy as np
import cv2
from .utils import show_image
# from .crop import remove_lateral_blacks

def warp_image(image, page_contour, border_pixels=0, show_step_by_step=False, show_overlay=True, opencv_version=True):
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(page_contour)
    center_box = rect[0]
    angle = rect[2]

    if angle > 80:
        angle = angle - 90
    angle = -angle  # Invert angle for correct rotation direction

    # ---- Crop SENZA rotazione (crop originale dal box) ----
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x0, y0, w0, h0 = cv2.boundingRect(box)
    x0 = max(0, int(x0 - border_pixels))
    y0 = max(0, int(y0 - border_pixels))
    h0 = h0 + int(border_pixels * 2)
    w0 = w0 + int(border_pixels * 2)
    crop_no_rotation = image[y0:y0+h0, x0:x0+w0]

    M = cv2.getRotationMatrix2D(center_box, angle, 1.0)

    # ---- Rotazione ----
    if opencv_version:
        M = cv2.getRotationMatrix2D(center_box, -angle, 1.0)
        # Rotazione con OpenCV (INTER_LANCZOS4 = best)
        # Calcola bbox della nuova immagine
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(image.shape[0] * sin + image.shape[1] * cos)
        new_h = int(image.shape[0] * cos + image.shape[1] * sin)
        
        # Aggiusta la matrice M per centrare il contenuto
        M[0, 2] += (new_w - image.shape[1]) / 2
        M[1, 2] += (new_h - image.shape[0]) / 2
        
        # Rotazione con dimensione più grande → niente taglio
        rotated_np = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    else:
        # Rotazione con pyvips (nohalo)
        height, width = image.shape[:2]
        bands = image.shape[2] if len(image.shape) == 3 else 1
        linear = image.reshape(height * width * bands)
        vips_image = pyvips.Image.new_from_memory(linear.tobytes(), width, height, bands, "uchar")

        angle_rad = np.radians(angle)
        a = np.cos(angle_rad)
        b = -np.sin(angle_rad)
        c = np.sin(angle_rad)
        d = np.cos(angle_rad)

        rotated = vips_image.affine([a, b, c, d], interpolate=pyvips.Interpolate.new("nohalo"))

        rotated_mem = rotated.write_to_memory()
        rotated_np = np.frombuffer(rotated_mem, dtype=np.uint8)
        rotated_np = rotated_np.reshape(rotated.height, rotated.width, rotated.bands).copy()

    # ---- Crop DOPO rotazione ----
    rotated_box = cv2.transform(np.array([box], dtype="float32"), M)[0]
    x, y, w, h = cv2.boundingRect(rotated_box)
    x = max(0, int(x - border_pixels))
    y = max(0, int(y - border_pixels))
    h = h + int(border_pixels * 2)
    w = w + int(border_pixels * 2)
    cropped = rotated_np[y:y+h, x:x+w]

    # ---- Show ----
    if show_step_by_step:
        if show_overlay:
            overlay = cropped.copy()
            overlay_contour = rotated_box.copy()
            overlay_contour[:, 0] -= x
            overlay_contour[:, 1] -= y
            cv2.drawContours(overlay, [overlay_contour.astype(np.int32)], -1, (0, 255, 0), 100)
            show_image(overlay, "Rotated and Cropped (with original contour)")
        else:
            show_image(cropped, "Rotated and Cropped")

    return cropped, crop_no_rotation
