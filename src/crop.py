import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import show_image



def remove_lateral_blacks(warped, warped_white, show_step_by_step=False, plot_corner_discontinuity=False):
    """
    Crop the image by removing the deepest black pixels found on the borders.

    This function processes a warped image to remove black borders by identifying
    the deepest black pixels along the edges and cropping them out.

    Args:
        warped (numpy.ndarray): The warped image to be cropped.
        warped_white (numpy.ndarray): A version of the warped image with white background.
        show_step_by_step (bool): If True, displays intermediate steps of the process.
        plot_corner_discontinuity (bool): If True, plots the corner discontinuity analysis.

    Returns:
        numpy.ndarray: The cropped image with black borders removed.

    Notes:
        - The function first converts the image to grayscale and applies a binary
          threshold to identify black areas.
        - It initializes cropping boundaries and iteratively finds the deepest black
          pixels from each side (top, bottom, left, right).
        - The function handles corner discontinuities by analyzing edge points and
          adjusting the crop accordingly.
        - If `plot_corner_discontinuity` is enabled, it visualizes the number of black
          pixels along each edge and the binary image.
        - The final cropped image is returned, with optional visualization of the
          cropping process if `show_step_by_step` is enabled.
    """
    binary, _ = get_binary_edges(warped, show_step_by_step=show_step_by_step)

    cropped, cropped_white = crop_black_borders(warped, warped_white, binary)

    if show_step_by_step:
        show_image(cropped, "Image without black band borders")

    binary, edges = get_binary_edges(cropped)

    crop_points = navigate_edges(binary, edges)

    binary, cropped, cropped_white = apply_cropping(crop_points, edges, binary, cropped, cropped_white)

    if plot_corner_discontinuity:
        plot_corner_discontinuity(binary)

    if show_step_by_step:
        show_image(cropped_white, "Cropped Image")

    return cropped_white

def plot_corner_discontinuity(binary):
    """
    Plot the corner discontinuity analysis of a binary image.

    This function visualizes the number of black pixels along each edge of the binary image
    and displays the binary image itself.

    Args:
        binary (numpy.ndarray): The binary image to analyze.

    Returns:
        None
    """
    fig, axs = plt.subplots(5, 1, figsize=(10, 30))

    # Plot top line values
    top_line_values = binary[0, :]
    axs[0].plot(top_line_values)
    axs[0].set_title(f'Top Line from left to right, number of black pixels: {np.sum(top_line_values == 0)}')

    # Plot bottom line values
    bottom_line_values = binary[binary.shape[0] - 1, :]
    axs[1].plot(bottom_line_values)
    axs[1].set_title(f'Bottom Line from left to right, number of black pixels: {np.sum(bottom_line_values == 0)}')

    # Plot left line values
    left_line_values = binary[:, 0]
    axs[2].plot(left_line_values)
    axs[2].set_title(f'Left Line from top to bottom, number of black pixels: {np.sum(left_line_values == 0)}')

    # Plot right line values
    right_line_values = binary[:, binary.shape[1] - 1]
    axs[3].plot(right_line_values)
    axs[3].set_title(f'Right Line from top to bottom, number of black pixels: {np.sum(right_line_values == 0)}')

    # Display the binary image
    axs[4].imshow(binary, cmap='gray')
    axs[4].set_title('Binary Image')

    plt.show()

def apply_cropping(crop_points, edges, binary, cropped, cropped_white):
    """
    Apply cropping to the images based on the detected crop points.

    Args:
        crop_points (dict): Dictionary containing cropping points and directions.
        edges (dict): Dictionary containing edge information.
        binary (numpy.ndarray): Binary image to be cropped.
        cropped (numpy.ndarray): Cropped image to be adjusted.
        cropped_white (numpy.ndarray): Cropped white image to be adjusted.

    Returns:
        tuple: Cropped binary, cropped, and cropped_white images.
    """
    for edge_name, edge_data in edges.items():
        if edge_name in crop_points:
            cropping_point = crop_points[edge_name]['point']
            cropping_direction = crop_points[edge_name]['direction']
            keep = crop_points[edge_name]['keep']
            if cropping_direction == 0:
                if keep == 'after':
                    binary = binary[cropping_point[0]:, :]
                    cropped = cropped[cropping_point[0]:, :]
                    cropped_white = cropped_white[cropping_point[0]:, :]
                else:
                    binary = binary[:cropping_point[0], :]
                    cropped = cropped[:cropping_point[0], :]
                    cropped_white = cropped_white[:cropping_point[0], :]
            elif cropping_direction == 1:
                if keep == 'after':
                    binary = binary[:, cropping_point[1]:]
                    cropped = cropped[:, cropping_point[1]:]
                    cropped_white = cropped_white[:, cropping_point[1]:]
                else:
                    binary = binary[:, :cropping_point[1]]
                    cropped = cropped[:, :cropping_point[1]]
                    cropped_white = cropped_white[:, :cropping_point[1]]

    return binary, cropped, cropped_white

def navigate_edges(binary, edges):
    """
    Convert the cropped image to binary and navigate its edges to find discontinuities and determine cropping points.

    Args:
        cropped (numpy.ndarray): Cropped image to analyze.

    Returns:
        dict: Crop points with their respective directions and keep criteria.
    """

    crop_points = {}

    for edge_name, edge_data in edges.items():
        x, y = edge_data['point']
        direction = edge_data['direction']
        if binary[x, y] == 255:
            discontinuities = np.array(edge_data['point'])
            for i in range(binary.shape[0]):
                point = x + i * direction[0]
                if binary[point, y] == 255:
                    discontinuities[0] = point
                else:
                    break
            for i in range(binary.shape[1]):
                point = y + i * direction[1]
                if binary[x, point] == 255:
                    discontinuities[1] = point
                else:
                    break

            linear_distance_from_edge = np.abs(np.array(edge_data['point']) - discontinuities)
            cropping_idx = np.argmin(linear_distance_from_edge)
            cropping_point = edge_data['point'].copy()
            cropping_point[cropping_idx] = discontinuities[cropping_idx]

            if direction[cropping_idx] == 1:
                keep = 'after'
            else:
                keep = 'before'

            crop_points[edge_name] = {'point': cropping_point, 'direction': cropping_idx, 'keep': keep}

    return crop_points

def get_binary_edges(cropped, show_step_by_step=False):
    """
    Convert the cropped image to a binary image and define edge points and directions.

    Args:
        cropped (numpy.ndarray): Cropped image to process.

    Returns:
        tuple: A binary image and a dictionary of edge points with directions.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if show_step_by_step:
        show_image(gray, "Gray Image")

    # Threshold the image to get a binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    if show_step_by_step:
        show_image(binary, "Binary Image")

    # Define edge points and directions
    top_left_edge = [0, 0]
    top_right_edge = [0, binary.shape[1] - 1]
    bottom_left_edge = [binary.shape[0] - 1, 0]
    bottom_right_edge = [binary.shape[0] - 1, binary.shape[1] - 1]

    edges = {
        'top-left': {'point': top_left_edge, 'direction': [1, 1]},
        'top-right': {'point': top_right_edge, 'direction': [1, -1]},
        'bottom-left': {'point': bottom_left_edge, 'direction': [-1, 1]},
        'bottom-right': {'point': bottom_right_edge, 'direction': [-1, -1]}
    }

    return binary, edges

def crop_black_borders(warped, warped_white, binary):
    """
    Crop the black borders from the warped image using the binary image.

    Args:
        warped (numpy.ndarray): Warped image to be cropped.
        warped_white (numpy.ndarray): Warped image with a white background to be cropped.
        binary (numpy.ndarray): Binary image used to detect black borders.

    Returns:
        tuple: Cropped versions of the warped and warped_white images.
    """
    # Initialize cropping boundaries
    top_black, bottom_black, left_black, right_black = 0, binary.shape[0], 0, binary.shape[1]

    # Find the topmost black row
    for row in range(binary.shape[0]):
        if np.all(binary[row, :] == 255):
            top_black = row
        else:
            break

    # Find the bottommost black row
    for row in range(binary.shape[0] - 1, -1, -1):
        if np.all(binary[row, :] == 255):
            bottom_black = row
        else:
            break

    # Find the leftmost black column
    for col in range(binary.shape[1]):
        if np.all(binary[:, col] == 255):
            left_black = col
        else:
            break

    # Find the rightmost black column
    for col in range(binary.shape[1] - 1, -1, -1):
        if np.all(binary[:, col] == 255):
            right_black = col
        else:
            break

    # Crop the images using the found boundaries
    cropped = warped[top_black+1:bottom_black-1, left_black+1:right_black-1]
    cropped_white = warped_white[top_black+1:bottom_black-1, left_black+1:right_black-1]

    return cropped, cropped_white
