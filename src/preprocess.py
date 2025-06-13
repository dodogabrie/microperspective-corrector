import cv2
import numpy as np
from .utils import show_image

def estimate_threshold(gray):
    """
    Stima una soglia binaria dinamica basata sulla luminosità media del bordo e del centro,
    utilizzando operazioni in float32 per ottimizzare velocità e uso di memoria.

    Assunzioni:
    - L'immagine in scala di grigi (gray) è di tipo uint8, con valori 0-255.
    - Il bordo (5% della dimensione minima) rappresenta lo sfondo.
    - Il centro (10% della dimensione minima) contiene l'oggetto principale.
    
    Parametri:
    - gray (np.ndarray): immagine in scala di grigi (dtype=uint8).

    Ritorna:
    - int: valore della soglia stimata (compreso tra 0 e 255).
    """
    h, w = gray.shape
    min_dim = min(h, w)
    border_size = int(min_dim * 0.05)  # 5% della dimensione minima
    center_size = int(min_dim * 0.1)  # 10% della dimensione minima

    # Calcola la media dei pixel dei quattro bordi usando float32 per ottimizzazione
    mean_top    = np.mean(gray[:border_size, :], dtype=np.float32)
    mean_bottom = np.mean(gray[-border_size:, :], dtype=np.float32)
    mean_left   = np.mean(gray[:, :border_size], dtype=np.float32)
    mean_right  = np.mean(gray[:, -border_size:], dtype=np.float32)
    border_mean = (mean_top + mean_bottom + mean_left + mean_right) / 4.0

    # Calcola la media della patch centrale
    cx, cy = w // 2, h // 2
    half_center = center_size // 2
    center_patch = gray[cy - half_center:cy + half_center, cx - half_center:cx + half_center]
    center_mean = np.mean(center_patch, dtype=np.float32)

    # Calcola la soglia come interpolazione lineare tra border_mean e center_mean
    alpha = 0.6  # Fattore di bilanciamento: più vicino a 1 dà maggiore importanza al centro
    threshold_val = border_mean + (center_mean - border_mean) * alpha

    # Limita il valore tra 0 e 255 e restituisce un int
    return int(np.clip(threshold_val, 0, 255))

def preprocess_image(image, show_step_by_step=False):
    """
    Converte l'immagine in scala di grigi, la sfoca e applica una soglia binaria dinamica.
    Mostra i passaggi intermedi se show_step_by_step è True.
    """
    # Converte l'immagine a scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show_step_by_step:
        show_image(gray, "Grayscale")

    # Calcola una dimensione del kernel basata sulla dimensione minima dell'immagine
    min_dim = min(gray.shape[:2])
    k = max(3, int((min_dim / 50) // 2 * 2 + 1))  # Kernel dispari, minimo 3x3
    k = min(k, 51)  # Limita la dimensione massima del kernel a 51
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    if show_step_by_step:
        show_image(blurred, f"Blurred (kernel={k}x{k})")

    # Calcola la soglia dinamica e applica la threshold
    threshold_val = estimate_threshold(blurred)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    if show_step_by_step:
        show_image(thresh, "Thresholded")

    return thresh
