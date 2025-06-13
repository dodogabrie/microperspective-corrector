import pyvips
import numpy as np
import cv2
from .utils import show_image

def warp_image(image, page_contour, border_pixels=0, show_step_by_step=False, show_overlay=True,
               border_value=(0, 0, 0), opencv_version=True):
    """
    Applica una trasformazione affine per raddrizzare una pagina rilevata nell'immagine,
    ruotandola in base all'orientamento e ritagliandola. Supporta OpenCV o pyvips per la rotazione.

    Args:
        image (np.ndarray): Immagine BGR di input.
        page_contour (np.ndarray): Contorno della pagina rilevato (Nx1x2).
        border_pixels (int): Margine extra da aggiungere al crop (prima e dopo rotazione).
        show_step_by_step (bool): Se True, mostra immagini intermedie.
        show_overlay (bool): Se True, disegna il contorno originale sulla regione ritagliata.
        border_value (tuple[int, int, int]): Colore di riempimento nei bordi (B, G, R).
        opencv_version (bool): Se True, usa OpenCV per la rotazione; altrimenti pyvips.

    Returns:
        cropped (np.ndarray): Immagine ritagliata e raddrizzata.
        crop_no_rotation (np.ndarray): Ritaglio rettangolare originale senza rotazione.
    """

    # Ottiene il rettangolo minimo che racchiude il contorno
    rect = cv2.minAreaRect(page_contour)
    center_box = rect[0]       # centro del rettangolo
    angle = rect[2]            # angolo in gradi

    # Corregge l’angolo se è quasi verticale
    if angle > 80:
        angle -= 90
    angle = -angle  # Inverte direzione della rotazione per OpenCV

    # Estrae il box e il crop prima della rotazione
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x0, y0, w0, h0 = cv2.boundingRect(box)
    x0 = max(0, int(x0 - border_pixels))
    y0 = max(0, int(y0 - border_pixels))
    h0 += int(border_pixels * 2)
    w0 += int(border_pixels * 2)
    crop_no_rotation = image[y0:y0+h0, x0:x0+w0]

    # Calcola matrice di rotazione (usata in entrambi i metodi)
    M = cv2.getRotationMatrix2D(center_box, -angle, 1.0)

    if opencv_version:
        # Calcola le nuove dimensioni dell’immagine dopo rotazione
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(image.shape[0] * sin + image.shape[1] * cos)
        new_h = int(image.shape[0] * cos + image.shape[1] * sin)

        # Aggiusta la traslazione per centrare l'immagine
        M[0, 2] += (new_w - image.shape[1]) / 2
        M[1, 2] += (new_h - image.shape[0]) / 2

        # Applica la rotazione con interpolazione Lanczos4 (alta qualità)
        rotated_np = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )

    else:
        # Rotazione tramite pyvips (nohalo), attorno all'origine
        height, width = image.shape[:2]
        bands = image.shape[2] if len(image.shape) == 3 else 1
        linear = image.reshape(height * width * bands)
        vips_image = pyvips.Image.new_from_memory(
            linear.tobytes(), width, height, bands, "uchar"
        )

        angle_rad = np.radians(angle)
        a, b = np.cos(angle_rad), -np.sin(angle_rad)
        c, d = np.sin(angle_rad),  np.cos(angle_rad)

        rotated = vips_image.affine([a, b, c, d], interpolate=pyvips.Interpolate.new("nohalo"))
        rotated_mem = rotated.write_to_memory()
        rotated_np = np.frombuffer(rotated_mem, dtype=np.uint8).reshape(rotated.height, rotated.width, rotated.bands).copy()

    # Trasforma il box originale con la matrice di rotazione per ritagliare il contenuto corretto
    rotated_box = cv2.transform(np.array([box], dtype="float32"), M)[0]
    x, y, w, h = cv2.boundingRect(rotated_box)
    x = max(0, int(x - border_pixels))
    y = max(0, int(y - border_pixels))
    w += int(border_pixels * 2)
    h += int(border_pixels * 2)
    cropped = rotated_np[y:y+h, x:x+w]

    # Visualizza il risultato se richiesto
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
