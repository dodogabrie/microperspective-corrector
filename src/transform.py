import pyvips
import numpy as np
import cv2
from .utils import show_image


def crop_image(image, box, border_pixels=0):
    """Crop image around a box with optional border padding."""
    x, y, w, h = cv2.boundingRect(box)
    x = max(0, int(x - border_pixels))
    y = max(0, int(y - border_pixels))
    w += int(border_pixels * 2)
    h += int(border_pixels * 2)
    return image[y:y+h, x:x+w]


def calculate_subject_intensity(gray_image, box, crop_offset=200):
    """
    Calculate the average intensity inside the detected box region.
    
    Args:
        gray_image: Grayscale image
        box: Box coordinates from contour detection
        crop_offset: Offset used when cropping the large region
        
    Returns:
        int: Average intensity value inside the box
    """
    box_coords = cv2.boundingRect(box)
    x_box, y_box, w_box, h_box = box_coords
    
    # Adjust box coordinates to cropped image coordinate system
    x_box_adj = x_box - (cv2.boundingRect(box)[0] - crop_offset)
    y_box_adj = y_box - (cv2.boundingRect(box)[1] - crop_offset)
    
    # Ensure coordinates are within image bounds
    h_img, w_img = gray_image.shape[:2]
    x_box_adj = max(0, min(x_box_adj, w_img))
    y_box_adj = max(0, min(y_box_adj, h_img))
    w_box = min(w_box, w_img - x_box_adj)
    h_box = min(h_box, h_img - y_box_adj)
    
    if w_box > 0 and h_box > 0:
        subject_region = gray_image[y_box_adj:y_box_adj+h_box, x_box_adj:x_box_adj+w_box]
        return int(np.mean(subject_region))
    else:
        # Fallback to center region if box coordinates are invalid
        center_h, center_w = h_img // 4, w_img // 4
        y_center = h_img // 2 - center_h // 2
        x_center = w_img // 2 - center_w // 2
        return int(np.mean(gray_image[y_center:y_center+center_h, x_center:x_center+center_w]))


def create_similarity_mask(gray_image, border_intensity, subject_intensity):
    """
    Create a binary mask where 1 indicates subject-like pixels and 0 indicates border-like pixels.
    
    Args:
        gray_image: Grayscale input image
        border_intensity: Expected intensity of border/background pixels
        subject_intensity: Expected intensity of subject/content pixels
        
    Returns:
        numpy.ndarray: Binary mask (0 or 1 values)
    """
    border_diff = np.abs(gray_image.astype(np.float32) - border_intensity)
    subject_diff = np.abs(gray_image.astype(np.float32) - subject_intensity)
    
    # Pixels closer to subject than to border get value 1
    return np.where(subject_diff < border_diff, 1.0, 0.0)


def find_mask_bounding_box(similarity_mask):
    """
    Find the bounding box of non-zero regions in the similarity mask.
    
    Args:
        similarity_mask: Binary mask where 1 indicates content pixels
        
    Returns:
        tuple: (x, y, w, h) bounding box coordinates, or None if no content found
    """
    non_zero_y, non_zero_x = np.nonzero(similarity_mask > 0.5)
    if non_zero_y.size == 0 or non_zero_x.size == 0:
        return None
    
    top = np.min(non_zero_y)
    bottom = np.max(non_zero_y)
    left = np.min(non_zero_x)
    right = np.max(non_zero_x)
    
    return left, top, right - left, bottom - top


def irregolar_border(image, box, border_value, step_by_step=False):
    """
    Detect the optimal crop box using intensity-based similarity analysis.
    
    This function creates a large crop around the detected box, applies blur to smooth
    details, then uses intensity differences to distinguish between content and background.
    
    Args:
        image: Input rotated image
        box: Detected box coordinates from contour detection
        border_pixels: Border padding (not used in current implementation)
        border_value: Expected background color from rotation
        step_by_step: Whether to show intermediate results
        
    Returns:
        tuple: (x, y, w, h, mask) crop coordinates in ORIGINAL image space and similarity mask, or None if failed
    """
    # Calculate the crop offset to convert back to original coordinates
    box_rect = cv2.boundingRect(box)
    crop_offset_x = max(0, box_rect[0] - 200)
    crop_offset_y = max(0, box_rect[1] - 200)
    
    # Create a large crop around the detected box for analysis
    crop_large = crop_image(image.copy(), box, 200)
    
    # Apply strong blur to focus on large regions rather than fine details
    kernel_size = max(21, min(crop_large.shape[:2]) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred_crop = cv2.GaussianBlur(crop_large, (kernel_size, kernel_size), 0)
    
    if step_by_step:
        show_image(blurred_crop, "Super Blurred Image", max_width=800, max_height=600)

    # Convert border_value to grayscale intensity
    if isinstance(border_value, tuple):
        border_intensity = int(np.mean(border_value))
    else:
        border_intensity = int(border_value)
    
    # Convert to grayscale for intensity analysis
    if len(blurred_crop.shape) == 3:
        gray = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = blurred_crop
    
    # Calculate subject intensity from the box region
    subject_intensity = calculate_subject_intensity(gray, box, crop_offset=200)
    
    # Create binary similarity mask
    similarity_mask = create_similarity_mask(gray, border_intensity, subject_intensity)
    
    if step_by_step:
        print(f"Border intensity: {border_intensity}, Subject intensity: {subject_intensity}")
        show_image(similarity_mask, "Similarity Mask", max_width=800, max_height=600)
    
    # Find bounding box of content regions
    bbox = find_mask_bounding_box(similarity_mask)
    if bbox is None:
        print("No content pixels found in the similarity mask.")
        return None
    
    x, y, w, h = bbox
    
    # Convert coordinates back to original image space
    x_original = x + crop_offset_x
    y_original = y + crop_offset_y
    
    if step_by_step:
        # Visualize the detected box with high contrast overlay
        box_vis = crop_large.copy()
        
        # Draw thick white border for contrast
        cv2.rectangle(box_vis, (x, y), (x + w, y + h), (255, 255, 255), 12)
        # Draw thinner colored rectangle on top
        cv2.rectangle(box_vis, (x, y), (x + w, y + h), (0, 255, 0), 6)
        
        # Add corner markers for better visibility
        corner_size = 20
        corner_thickness = 8
        corners = [
            (x, y), (x + w, y), (x, y + h), (x + w, y + h)
        ]
        for corner_x, corner_y in corners:
            # White background for contrast
            cv2.circle(box_vis, (corner_x, corner_y), corner_size, (255, 255, 255), corner_thickness + 4)
            # Colored marker on top
            cv2.circle(box_vis, (corner_x, corner_y), corner_size, (255, 0, 0), corner_thickness)
        
        show_image(box_vis, "Detected Box from Similarity Mask (Cropped View)", max_width=800, max_height=600)
        print(f"Detected box in cropped space: x={x}, y={y}, w={w}, h={h}")
        print(f"Detected box in original space: x={x_original}, y={y_original}, w={w}, h={h}")
        print(f"Crop offset: ({crop_offset_x}, {crop_offset_y})")
    
    return x_original, y_original, w, h, similarity_mask


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
    crop_coords = irregolar_border(rotated_np.copy(), rotated_box, border_value, show_step_by_step)
    
    if crop_coords is not None:
        x, y, w, h, _ = crop_coords
        # Add border padding while ensuring we don't go outside image bounds
        img_height, img_width = rotated_np.shape[:2]
        x = max(0, int(x - border_pixels))
        y = max(0, int(y - border_pixels))
        w = min(img_width - x, w + int(border_pixels * 2))
        h = min(img_height - y, h + int(border_pixels * 2))
    else:
        # Fallback to original bounding rect if irregolar_border fails
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
            cv2.drawContours(overlay, [overlay_contour.astype(np.int32)], -1, (0, 255, 0), 15)
            show_image(overlay, "Rotated and Cropped (with original contour)")
        else:
            show_image(cropped, "Rotated and Cropped")

    return cropped, crop_no_rotation
