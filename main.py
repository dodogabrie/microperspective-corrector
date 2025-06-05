import argparse
import os
from edge_detection import process_tiff
from src.spinner import Spinner
from report import generate_html_report

def main(input_dir, output_dir, border_pixels=2, verbose=True):
    """
    Process TIFF images from the input directory and save the processed images to the output directory.

    This function iterates over all TIFF files in the specified input directory, processes each file
    using the `process_tiff` function, and saves the processed images to the specified output directory.
    It also displays a progress spinner and updates the progress information in the console.

    Args:
        input_dir (str): Directory containing input TIFF files.
        output_dir (str): Directory to save processed TIFF files.

    Returns:
        None

    Notes:
        - The function ensures that the output directory exists before processing.
        - It uses a spinner to provide visual feedback on the processing progress.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of TIFF files
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
    total_files = len(tiff_files)

    # Create and start the spinner
    if verbose:
        spinner = Spinner(total_files)
        spinner.start()

    # Process all .tiff files in the input directory
    for i, filename in enumerate(tiff_files):
        input_path = os.path.join(input_dir, filename)
        output_tiff_path = os.path.join(output_dir, filename)

        # Process the selected file
        process_tiff(input_path, output_tiff_path, output_path_thumb=None, border_pixels=border_pixels, show_step_by_step=False, show_before_after=False)

        # Update progress
        if verbose:
            spinner.update_progress(i, filename)

    # Stop the spinner
    if verbose:
        spinner.stop()

    generate_html_report('report.html')
    print("\nProcessing complete!")


if __name__ == "__main__":
    """
    Command-line interface for processing TIFF images.

    This script processes all TIFF images in a specified input directory and saves the processed
    images to a specified output directory. It uses command-line arguments to specify the directories.

    Usage:
        python main.py <input_dir> <output_dir>

    Arguments:
        input_dir: Directory containing input TIFF files.
        output_dir: Directory to save processed TIFF files.
    """
    parser = argparse.ArgumentParser(description="Process TIFF images to remove black borders and warp them.")
    parser.add_argument("input_dir", type=str, help="Directory containing input TIFF files.")
    parser.add_argument("output_dir", type=str, help="Directory to save processed TIFF files.")
    parser.add_argument("-b", "--border", type=int, default=20, help="Number of pixels for the external border.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, border_pixels=args.border, verbose=args.verbose)
