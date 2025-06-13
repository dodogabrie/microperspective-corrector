import cv2
import numpy as np
from .utils import show_image


def rgb_to_gray_from_tuple(bgr):
    """
    Converte una tripla (B, G, R) in valore scala di grigi usando la formula BT.601.

    Args:
        bgr (tuple[float, float, float]): Valori medi BGR.

    Returns:
        float: Valore di grigio corrispondente.
    """
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def estimate_threshold_and_border_rgb(image, gray_blurred):
    """
    Calcola una soglia binaria dinamica e il valore medio RGB dei bordi dell'immagine.

    Args:
        image (np.ndarray): Immagine BGR originale (uint8).
        gray_blurred (np.ndarray): Immagine sfocata in scala di grigi (uint8).

    Returns:
        int: Valore di soglia stimato per la binarizzazione (0–255).
        tuple[float, float, float]: Valore medio (B, G, R) dei bordi in float32.
    """
    h, w = gray_blurred.shape
    min_dim = min(h, w)
    border = int(min_dim * 0.05)       # 5% per il bordo
    center = int(min_dim * 0.1) // 2   # 10% per il centro

    # Estrai bordi e concatena
    top = image[:border]
    bottom = image[-border:]
    left = image[:, :border]
    right = image[:, -border:]

    # Calcola media RGB dei bordi
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0).astype(np.float32)
    border_rgb = tuple(border_pixels.mean(axis=0))  # B, G, R

    # Converti media RGB in grigio
    border_gray = rgb_to_gray_from_tuple(border_rgb)

    # Calcola media nel centro dell'immagine grigia
    cx, cy = w // 2, h // 2
    center_patch = gray_blurred[cy - center:cy + center, cx - center:cx + center]
    center_mean = center_patch.mean(dtype=np.float32)

    # Interpolazione pesata
    alpha = 0.6
    threshold_val = int(np.clip(border_gray + (center_mean - border_gray) * alpha, 0, 255))

    return threshold_val, tuple(int(round(c)) for c in border_rgb)


def preprocess_image(image, show_step_by_step=False):
    """
    Converte l'immagine in scala di grigi, la sfoca, calcola soglia dinamica
    basata sui valori medi dei bordi e binarizza. Restituisce anche il valore RGB medio dei bordi.

    Args:
        image (np.ndarray): Immagine BGR originale (uint8).
        show_step_by_step (bool): Se True, mostra i passaggi.

    Returns:
        np.ndarray: Immagine binarizzata (uint8, 0 o 255).
        tuple[float, float, float]: Media (B, G, R) dei bordi in float32.
    """
    # Converte in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_step_by_step:
        show_image(gray, "Grayscale")

    # Applica blur adattivo in base alla dimensione minima
    min_dim = min(gray.shape[:2])
    k = max(3, int((min_dim / 50) // 2 * 2 + 1))  # Kernel dispari, minimo 3
    k = min(k, 51)  # Massimo 51
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    if show_step_by_step:
        show_image(blurred, f"Blurred (kernel={k}x{k})")

    # Calcola soglia e valore RGB del bordo
    threshold_val, border_rgb = estimate_threshold_and_border_rgb(image, blurred)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    if show_step_by_step:
        show_image(thresh, f"Thresholded (th={threshold_val})")

    return thresh, border_rgb
