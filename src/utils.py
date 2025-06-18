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
        image_description (str): Description of the image.

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


def save_outputs(original, processed, output_path_tiff, output_path_thumb=None, copied=False, output_no_cropped=None, original_path=None):
    """
    Save the processed TIFF image, a reduced JPG thumbnail, and the quality evaluation JSON.
    Now preserves metadata and uses lossless compression.
    """
    if copied:
        processed = np.zeros_like(original)
    
    # Salva l'immagine processata con metadati preservati e qualità lossless
    if original_path:
        save_image_with_metadata(processed, output_path_tiff, original_path)
    else:
        # Salvataggio lossless anche senza metadati
        if output_path_tiff.lower().endswith(('.tiff', '.tif')):
            cv2.imwrite(output_path_tiff, processed)  # TIFF è già lossless in OpenCV
        elif output_path_tiff.lower().endswith('.png'):
            cv2.imwrite(output_path_tiff, processed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif output_path_tiff.lower().endswith(('.jpg', '.jpeg')):
            cv2.imwrite(output_path_tiff, processed, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(output_path_tiff, processed)

    # Per il thumbnail, puoi comunque usare compressione (è un'anteprima)
    if output_path_thumb:
        # Riduci dimensioni per thumbnail
        height, width = processed.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            thumbnail = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            thumbnail = processed.copy()
        
        # Salva thumbnail con compressione moderata (è solo un'anteprima)
        cv2.imwrite(output_path_thumb, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return thumbnail
    
    return processed

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