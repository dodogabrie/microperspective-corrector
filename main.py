import argparse
import os
import sys

from edge_detection import process_tiff
from src.spinner import Spinner
from report import generate_html_report

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
         output_path_thumb=None, image_input_format=None):
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

    Returns:
        None

    Note:
        - La struttura delle sottocartelle viene replicata nella cartella di output.
        - Vengono processati solo file .tif, .tiff, .jpg, .jpeg.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Trova tutte le immagini ricorsivamente
    if image_input_format:
        image_input_format = image_input_format.lower()
        if image_input_format not in ['tif', 'tiff', 'jpg', 'jpeg']:
            raise ValueError("Invalid image input format. Use 'tif', 'tiff', 'jpg', or 'jpeg'.")
        format = [f'.{image_input_format}']
    else:
        format = ['.tif', '.tiff', '.jpg', '.jpeg']

    image_files = find_images_recursive(input_dir, format=format)
    total_files = len(image_files)

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
        process_tiff(
            input_path,
            output_path,
            output_path_thumb=output_path_thumb,
            border_pixels=border_pixels,
            show_step_by_step=False,
            show_before_after=False,
        )

        # Aggiorna lo spinner
        if verbose:
            spinner.update_progress(i, rel_path)

    # Ferma lo spinner
    if verbose:
        spinner.stop()

    generate_html_report('report.html')
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-t", "--output_thumb", type=str, default=None, help="Path to save reduced thumbnails (optional).")
    parser.add_argument("-f", "--image-input-format", type=str, default="tif", help="Input image format (tif/jpg).")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, border_pixels=args.border, 
         verbose=args.verbose, output_path_thumb=args.output_thumb, 
         image_input_format=args.image_input_format)
