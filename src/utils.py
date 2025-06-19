import cv2
import os
import numpy as np
import json
import time
from .quality_evaluation import evaluate_quality
from PIL import Image


DEFAULT_ERROR = {
    'sharpness': 0.1,
    'entropy': 0.1,
    'edge_density': 0.1,
    'residual_skew_angle': 0.1,
}

def show_image(image, title="Image", max_width=1280, max_height=720, file_path=None):
    """
    Resize and display an image with a title. Waits for a key press to close.

    Args:
        image (numpy.ndarray): Image to display.
        title (str): Window title for the displayed image.
        max_width (int): Maximum width for resizing.
        max_height (int): Maximum height for resizing.
        file_path (str): Path to save the image.

    Returns:
        None
    """
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1)
    resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

    if not file_path:
        cv2.imshow(title, resized_image)
        cv2.waitKey(0)

    # this is for educational purposes
    if file_path:
        cv2.imwrite(file_path, resized_image)


def load_image(image_path):
    """Load the input image from the given path."""
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


def save_image_with_metadata(image_array, output_path, original_path):
    """
    Salva un'immagine preservando i metadati EXIF senza perdita di qualità.
    
    Args:
        image_array (np.ndarray): Array dell'immagine da salvare (BGR format)
        output_path (str): Path dove salvare l'immagine
        original_path (str): Path dell'immagine originale (per i metadati)
    """
    try:
        # Converti da BGR (OpenCV) a RGB (PIL)
        if len(image_array.shape) == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_array
            
        # Crea immagine PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Determina il formato basandosi sull'estensione
        output_ext = os.path.splitext(output_path)[1].lower()
        
        # Carica metadati dall'originale
        try:
            with Image.open(original_path) as original:
                exif_bytes = original.info.get('exif')
                
                if output_ext in ['.tiff', '.tif']:
                    # TIFF: Lossless compression
                    if exif_bytes:
                        pil_image.save(output_path, format='TIFF', compression='lzw', exif=exif_bytes)
                    else:
                        pil_image.save(output_path, format='TIFF', compression='lzw')
                        
                elif output_ext in ['.png']:
                    # PNG: Lossless
                    pnginfo = original.info.copy()
                    # PNG non supporta EXIF direttamente, usa text metadata
                    pil_image.save(output_path, format='PNG', pnginfo=pnginfo)
                    
                elif output_ext in ['.jpg', '.jpeg']:
                    # JPEG: Usa qualità 100 (minima compressione)
                    if exif_bytes:
                        pil_image.save(output_path, format='JPEG', quality=100, exif=exif_bytes, optimize=False, subsampling=0)
                    else:
                        pil_image.save(output_path, format='JPEG', quality=100, optimize=False, subsampling=0)
                else:
                    # Default: formato auto-rilevato
                    if exif_bytes:
                        pil_image.save(output_path, exif=exif_bytes)
                    else:
                        pil_image.save(output_path)
                return
                        
        except Exception as e:
            print(f"Warning: Could not preserve EXIF data: {e}")
            
        # Fallback: salva senza metadati ma lossless
        if output_ext in ['.tiff', '.tif']:
            pil_image.save(output_path, format='TIFF', compression='lzw')
        elif output_ext in ['.png']:
            pil_image.save(output_path, format='PNG')
        elif output_ext in ['.jpg', '.jpeg']:
            pil_image.save(output_path, format='JPEG', quality=100, optimize=False, subsampling=0)
        else:
            pil_image.save(output_path)
        
    except Exception as e:
        print(f"Error saving with PIL, falling back to OpenCV: {e}")
        # Fallback a OpenCV con massima qualità
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            cv2.imwrite(output_path, image_array, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif output_path.lower().endswith('.png'):
            cv2.imwrite(output_path, image_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(output_path, image_array)


def save_outputs(original, processed, output_path_tiff, output_path_thumb=None, copied=False, output_no_cropped=None, original_path=None):
    """
    Save the processed TIFF image, a reduced JPG thumbnail, and the quality evaluation JSON.
    Always saves both original and processed images in the thumbnail, and always saves the quality file.
    Now also preserves metadata from original images.
    """
    if copied:
        processed = np.zeros_like(original)  # If copied, processed is an empty image
    
    # Save the processed TIFF with metadata preservation
    if original_path:
        save_image_with_metadata(processed, output_path_tiff, original_path)
    else:
        # Fallback: salvataggio lossless senza metadati
        if output_path_tiff.lower().endswith(('.tiff', '.tif')):
            cv2.imwrite(output_path_tiff, processed)
        elif output_path_tiff.lower().endswith('.png'):
            cv2.imwrite(output_path_tiff, processed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif output_path_tiff.lower().endswith(('.jpg', '.jpeg')):
            cv2.imwrite(output_path_tiff, processed, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(output_path_tiff, processed)

    # Ensure both images have the same height
    if original.shape[0] != processed.shape[0]:
        height = min(original.shape[0], processed.shape[0])
        original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        processed = cv2.resize(processed, (int(processed.shape[1] * height / processed.shape[0]), height))

    # Ensure both images have the same type
    if original.shape[2] != processed.shape[2]:
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # --- SEPARAZIONE VISIVA ---
    sep_width = 100  # larghezza separatore
    height = original.shape[0]
    separator = 0 * np.ones((height, sep_width, 3), dtype=np.uint8)  # grigio chiaro

    # Concatenate original, separator, processed
    concatenated_image = cv2.hconcat([original, separator, processed])

    resize_val = 500
    height, width = concatenated_image.shape[:2]
    thumbnail = cv2.resize(concatenated_image, (resize_val, int(resize_val * height / width)))

    # Se non viene passato output_path_thumb, salva in una cartella tmp accanto all'output tiff
    if not output_path_thumb:
        output_dir = os.path.dirname(output_path_tiff)
        output_path_thumb = os.path.join(output_dir, 'tmp')
        os.makedirs(output_path_thumb, exist_ok=True)

    # Extract the base filename and add timestamp for proper sorting
    base_filename = os.path.basename(output_path_tiff)
    name_without_ext = os.path.splitext(base_filename)[0]
    timestamp = str(int(time.time() * 1000))  # milliseconds timestamp
    thumbnail_filename = f"{timestamp}_{name_without_ext}.jpg"
    
    # Salva thumbnail con gestione errori migliorata
    thumbnail_full_path = os.path.join(output_path_thumb, thumbnail_filename)
    try:
        success = cv2.imwrite(thumbnail_full_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            # Fallback con PIL
            thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            pil_thumbnail = Image.fromarray(thumbnail_rgb)
            pil_thumbnail.save(thumbnail_full_path, format='JPEG', quality=85)
    except Exception as e:
        print(f"Warning: Failed to save thumbnail: {e}")

    # --- Calcola e salva la valutazione della qualità ---
    if output_no_cropped is not None:
        image_to_compare = output_no_cropped
    else:
        # If no uncropped original is provided, use the processed image
        image_to_compare = original 

    quality = evaluate_quality(image_to_compare, processed)

    quality_dir = os.path.join(os.path.dirname(output_path_thumb), 'quality')
    quality_dir = os.path.abspath(quality_dir)
    os.makedirs(quality_dir, exist_ok=True)
    quality_filename = os.path.splitext(thumbnail_filename)[0] + '.json'
    quality_path = os.path.join(quality_dir, quality_filename)
    
    def to_python_type(obj):
        if isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    quality_py = to_python_type(quality)
    with open(quality_path, 'w', encoding='utf-8') as f:
        json.dump(quality_py, f, indent=2, ensure_ascii=False)

    return thumbnail