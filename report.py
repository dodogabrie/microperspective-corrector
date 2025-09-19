import os

# import shutil


def generate_html_report(report_file):
    """
    Generate an HTML report with thumbnails of processed TIFF images and their filenames.

    Args:
        report_file (str): Path to save the generated HTML report.

    Returns:
        None
    """

    output_dir = "tmp"

    # Start the HTML content
    html_content = """
    <html>
    <head>
        <title>Miniature processate</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .thumbnail { margin: 10px; display: inline-block; }
            .thumbnail img { width: 400px; height: auto; }
            .thumbnail p { text-align: center; }
        </style>
    </head>
    <body>
        <h1>Report di immagini processate</h1>
        <p>Questa pagina contiene le miniature delle immagini processate. Consente un rapido controllo sulla qualità del cropping e delle rotazioni applicate dall'automatismo.</p>
        <div>
    """
    if not os.path.exists(output_dir):
        print(f"Error: The directory {output_dir} does not exist.")
        return

    files = sorted(os.listdir(output_dir))

    # Iterate over the thumbnails in the output directory
    for filename in files:
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(output_dir, filename)
            html_content += f"""
            <div class="thumbnail">
                <img src="{file_path}" alt="{filename}">
                <p>{filename.replace(".jpg", "")}</p>
            </div>
            """

    # Close the HTML content
    html_content += """
        </div>
    </body>
    </html>
    """

    # Write the HTML content to the report file
    with open(report_file, "w") as f:
        f.write(html_content)

    print(f"HTML report generated: {report_file}")
