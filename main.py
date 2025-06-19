import argparse
import os
import sys
import json
import time
from datetime import datetime

# Aggiungo al path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

sys.path.insert(0, current_dir)  # Per importare edge_detection, report, etc.
sys.path.insert(0, src_dir)      # Per i moduli src.* usati in edge_detection

from edge_detection import process_tiff
from src.spinner import Spinner
from report import generate_html_report

def get_file_size_gb(file_path):
    """Get file size in GB."""
    return os.path.getsize(file_path) / (1024**3)

def write_info_json(output_dir, info_data):
    """Write info.json file to output directory."""
    info_path = os.path.join(output_dir, 'info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)

def find_images_recursive(input_dir, format=['.tif', '.tiff', '.jpg', '.jpeg']):
    """
    Trova tutti i file tif/tiff/jpg/jpeg ricorsivamente nella directory di input e nelle sue sottocartelle.

    Args:
        input_dir (str): Directory di input.

    Returns:
        list: Lista dei percorsi completi dei file immagine trovati.
    """
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in format:
                image_files.append(os.path.join(root, f))
    return image_files

def main(input_dir, output_dir, border_pixels=1000, verbose=True, 
         output_path_thumb=None, image_input_format=None, show_step_by_step=False, use_compression=True):
    """
    Process images from the input directory (recursively) and save the processed images to the output directory,
    preserving the folder structure.

    Args:
        input_dir (str): Directory contenente le immagini di input (anche in sottocartelle).
        output_dir (str): Directory dove salvare le immagini elaborate (struttura replicata).
        border_pixels (int): Numero di pixel per il bordo esterno.
        verbose (bool): Se True, mostra avanzamento.
        output_path_thumb (str): Percorso per salvare le miniature ridotte (opzionale).
        image_input_format (str): Formato delle immagini di input (tif/jpg).
        use_compression (bool): Se True, usa compressione LZW per file TIFF (default: True).

    Returns:
        None

    Note:
        - La struttura delle sottocartelle viene replicata nella cartella di output.
        - Vengono processati solo file .tif, .tiff, .jpg, .jpeg.
    """
    print('Starting image processing...')
    start_time = time.time()
    start_datetime = datetime.now().isoformat()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Trova tutte le immagini ricorsivamente
    if image_input_format:
        image_input_format = image_input_format.lower()
        if image_input_format not in ['tif', 'tiff', 'jpg', 'jpeg']:
            raise ValueError("Invalid image input format. Use 'tif', 'tiff', 'jpg', or 'jpeg'.")
        format = [f'.{image_input_format}']
        if format == ['.tiff']:
            format.append('.tif')
        elif format == ['.tif']:
            format.append('.tiff')
        elif format == ['.jpg']:
            format.append('.jpeg')
        elif format == ['.jpeg']:
            format.append('.jpg')
    else:
        format = ['.tif', '.tiff', '.jpg', '.jpeg']

    image_files = find_images_recursive(input_dir, format=format)
    total_files = len(image_files)
    
    # Calculate total size
    total_size_gb = sum(get_file_size_gb(f) for f in image_files)
    
    # Determine primary format
    formats = {}
    for f in image_files:
        ext = os.path.splitext(f)[1].lower()
        formats[ext] = formats.get(ext, 0) + 1
    primary_format = max(formats.keys(), key=formats.get) if formats else 'unknown'
    
    # Create initial info.json
    info_data = {
        "total_images": total_files,
        "processed": 0,
        "successful_metadata_preservation": 0,
        "failed_metadata_preservation": 0,
        "primary_format": primary_format.upper().replace('.', ''),
        "all_formats": formats,
        "total_size_gb": round(total_size_gb, 3),
        "compression_enabled": use_compression,
        "start_time": start_datetime,
        "duration_seconds": None,
        "status": "processing"
    }
    
    write_info_json(output_dir, info_data)

    # Se non specificato, salva le thumb in output/thumbs
    if output_path_thumb is None:
        output_path_thumb = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../output/thumbs'))
    os.makedirs(output_path_thumb, exist_ok=True)

    # Crea e avvia lo spinner se richiesto
    if verbose:
        spinner = Spinner(total_files)
        spinner.start()
    

    for i, input_path in enumerate(image_files):
        # Calcola il percorso relativo rispetto alla input_dir
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        output_folder = os.path.dirname(output_path)
        os.makedirs(output_folder, exist_ok=True)

        # Processa il file
        thumbnail = process_tiff(
            input_path,
            output_path,
            output_path_thumb=output_path_thumb,
            border_pixels=border_pixels,
            show_step_by_step=show_step_by_step,
            show_before_after=False,
            use_compression=use_compression
        )

        # Check if metadata was preserved (read from quality JSON if available)
        quality_dir = os.path.join(output_path_thumb, 'quality')
        if os.path.exists(quality_dir):
            quality_files = [f for f in os.listdir(quality_dir) if f.endswith('.json')]
            if quality_files:
                latest_quality = sorted(quality_files)[-1]
                quality_path = os.path.join(quality_dir, latest_quality)
                try:
                    with open(quality_path, 'r') as f:
                        quality_data = json.load(f)
                        if 'metadata_comparison' in quality_data:
                            if quality_data['metadata_comparison'].get('metadata_preserved', False):
                                info_data["successful_metadata_preservation"] += 1
                            else:
                                info_data["failed_metadata_preservation"] += 1
                except:
                    pass

        # Aggiorna lo spinner
        if verbose:
            spinner.update_progress(i, rel_path)
            
        # Update progress in info.json
        info_data["processed"] = i + 1
        write_info_json(output_dir, info_data)

    # Ferma lo spinner
    if verbose:
        spinner.stop()

    # Update final info.json with completion data
    end_time = time.time()
    duration_seconds = round(end_time - start_time, 2)
    
    info_data["duration_seconds"] = duration_seconds
    info_data["status"] = "completed"
    info_data["end_time"] = datetime.now().isoformat()
    info_data["metadata_preservation_rate"] = round(
        info_data["successful_metadata_preservation"] / max(1, info_data["processed"]) * 100, 2
    ) if info_data["processed"] > 0 else 0
    
    write_info_json(output_dir, info_data)

    # generate_html_report('report.html')
    print("\nProcessing complete!")

if __name__ == "__main__":
    """
    Command-line interface for processing TIFF/JPG images.

    This script processes all TIFF/JPG images in a specified input directory (recursively)
    and saves the processed images to a specified output directory, preserving the folder structure.

    Usage:
        python main.py <input_dir> <output_dir>

    Arguments:
        input_dir: Directory containing input images (tif/tiff/jpg/jpeg, also in subfolders).
        output_dir: Directory to save processed images (structure will be replicated).
    """
    parser = argparse.ArgumentParser(description="Process images to remove black borders and warp them (recursive, preserves folder structure).")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images.")
    parser.add_argument("-b", "--border", type=int, default=100, help="Number of pixels for the external border.")
    parser.add_argument("-s", "--show_step_by_step", default=False, action="store_true", help="Show step-by-step processing (for debugging).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-t", "--output_thumb", type=str, default=None, help="Path to save reduced thumbnails (optional).")
    parser.add_argument("-f", "--image-input-format", type=str, default="tif", help="Input image format (tif/jpg).")
    parser.add_argument("--no-compression", action="store_true", help="Disable LZW compression for TIFF files (default: compression enabled).")

    args = parser.parse_args()
    
    # thumb directory
    output_dir = args.output_dir
    output_path_thumb = args.output_thumb
    if output_path_thumb is None:
        output_path_thumb = os.path.join(output_dir, 'thumb') # Default to output_dir + thumb if not specified

    main(args.input_dir, args.output_dir, border_pixels=args.border, 
         verbose=args.verbose, output_path_thumb=output_path_thumb,
         image_input_format=args.image_input_format, 
         show_step_by_step=args.show_step_by_step,
         use_compression=not args.no_compression)
