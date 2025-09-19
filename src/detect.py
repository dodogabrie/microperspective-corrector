import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line as skimage_line

from .utils import show_image


def plot_contour_side_distances(approx_contour):
    """
    Visualizza le distanze medie dei lati di un contorno approssimato con matplotlib.

    Parameters:
        approx_contour (numpy.ndarray): Contorno approssimato (tipicamente 4 punti per una pagina)
    """
    # Converti il contour in formato (n, 2)
    points = approx_contour.reshape(-1, 2)

    # Calcola le distanze dei lati
    side_distances = []
    side_labels = []

    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        distance = np.linalg.norm(points[next_i] - points[i])
        side_distances.append(distance)
        side_labels.append(f"Lato {i + 1}-{next_i + 1}")

    # Calcola la distanza media
    mean_distance = np.mean(side_distances)

    # Crea il grafico
    plt.figure(figsize=(10, 6))

    # Subplot 1: Grafico a barre delle distanze
    plt.subplot(1, 2, 1)
    bars = plt.bar(
        side_labels,
        side_distances,
        color=["red", "green", "blue", "orange"][: len(side_distances)],
    )
    plt.axhline(
        y=mean_distance,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Media: {mean_distance:.2f}",
    )
    plt.title("Distanze dei Lati del Contorno")
    plt.ylabel("Distanza (pixel)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Aggiungi valori sopra le barre
    for i, (bar, distance) in enumerate(zip(bars, side_distances)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mean_distance * 0.02,
            f"{distance:.1f}",
            ha="center",
            va="bottom",
        )

    # Subplot 2: Visualizzazione del contorno
    plt.subplot(1, 2, 2)
    # Chiudi il poligono per la visualizzazione
    closed_points = np.vstack([points, points[0]])
    plt.plot(closed_points[:, 0], closed_points[:, 1], "b-o", linewidth=2, markersize=8)

    # Etichetta i punti
    for i, point in enumerate(points):
        plt.annotate(
            f"P{i + 1}",
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Etichetta i lati con le loro distanze
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        mid_point = (points[i] + points[next_i]) / 2
        plt.annotate(
            f"{side_distances[i]:.1f}",
            mid_point,
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

    plt.title("Contorno con Distanze dei Lati")
    plt.xlabel("X (pixel)")
    plt.ylabel("Y (pixel)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.gca().invert_yaxis()  # Inverti Y per coordinare immagine

    plt.tight_layout()
    plt.show()

    # Stampa statistiche
    print("\n=== STATISTICHE CONTORNO ===")
    print(f"Numero di lati: {len(side_distances)}")
    print(f"Distanza media: {mean_distance:.2f} pixel")
    print(f"Distanza minima: {min(side_distances):.2f} pixel")
    print(f"Distanza massima: {max(side_distances):.2f} pixel")
    print(f"Deviazione standard: {np.std(side_distances):.2f} pixel")

    for i, distance in enumerate(side_distances):
        print(f"  {side_labels[i]}: {distance:.2f} pixel")


def contour_side_intensity(approx_contour, gray_image, show_plot=False):
    """
    Calcola l'intensità media dei pixel (scala di grigi) e l'inclinazione di ciascun lato del contorno approssimato.
    Associa ogni lato a centro o background e ritorna:
      - inclinazione media pesata (con segno, in gradi)
      - inclinazione assoluta media pesata (in gradi)

    Parameters:
        approx_contour (numpy.ndarray): Contorno approssimato (tipicamente 4 punti per una pagina)
        gray_image (numpy.ndarray): Immagine in scala di grigi da cui estrarre i valori di intensità
        show_plot (bool): Se True, mostra i plot. Default: False
    Returns:
        mean_weighted_angle (float): media pesata degli angoli rispetto OX (con segno, in gradi)
        mean_weighted_inclination (float): media pesata delle inclinazioni (in gradi, sempre positiva)
        side_angles (list): lista degli angoli rispetto OX (con segno, in gradi)
        side_inclinations (list): lista delle inclinazioni (in gradi, sempre positiva)
        side_assoc (list): lista delle etichette centro/background
    """
    points = approx_contour.reshape(-1, 2)
    side_intensities = []
    side_labels = []
    side_angles = []
    side_inclinations = []
    for i in range(len(points)):
        next_i = (i + 1) % len(points)
        x0, y0 = points[i]
        x1, y1 = points[next_i]
        rr, cc = skimage_line(int(y0), int(x0), int(y1), int(x1))
        rr = np.clip(rr, 0, gray_image.shape[0] - 1)
        cc = np.clip(cc, 0, gray_image.shape[1] - 1)
        pixel_values = gray_image[rr, cc]
        mean_intensity = np.mean(pixel_values)
        side_intensities.append(mean_intensity)
        side_labels.append(f"Lato {i + 1}-{next_i + 1}")
        # Calcola angolo rispetto all'asse orizzontale (OX)
        dx = x1 - x0
        dy = y1 - y0
        angle = np.degrees(np.arctan2(dy, dx))
        angle = (angle + 180) % 180 - 90  # range [-90, 90]
        side_angles.append(angle)
        # Inclinazione rispetto agli assi (0° o 90°)
        signed_candidate_array = np.array([angle, angle - 90, angle + 90])
        candidate_array = np.array([abs(angle), abs(angle - 90), abs(angle + 90)])
        min_index = np.argmin(candidate_array)
        side_inclinations.append(signed_candidate_array[min_index])
    # Calcola media dentro e fuori il box
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.drawContours(mask, [approx_contour], -1, 255, -1)
    inside_mean = np.mean(gray_image[mask == 255])
    outside_mean = np.mean(gray_image[mask == 0])
    # Associa centro/background
    side_assoc = []
    for intensity in side_intensities:
        if abs(intensity - inside_mean) < abs(intensity - outside_mean):
            side_assoc.append("centro")
        else:
            side_assoc.append("background")
    # Pesi: centro=1, background=0.2
    dict_weights = {"centro": 1.0, "background": 0.2}
    weights = [1.0 if assoc == "centro" else 0.2 for assoc in side_assoc]
    mean_weighted_inclination = np.average(side_inclinations, weights=weights)
    # Ordina per intensità decrescente
    sorted_idx = np.argsort(side_intensities)[::-1]
    if show_plot:
        print("\n=== CLASSIFICA LATI PER INTENSITÀ MEDIA ===")
        print("Idx | Intensità | Angolo rispetto OX | Inclinazione (°) | Associazione")
        print("----|-----------|--------------------|------------------|-------------")
        for rank, idx in enumerate(sorted_idx, 1):
            print(
                f"{rank:>3} | {side_intensities[idx]:9.1f} | {side_angles[idx]:17.1f}° | {side_inclinations[idx]:16.1f}° | {side_assoc[idx]}"
            )
    if show_plot:
        plt.figure(figsize=(12, 6))
        # Subplot 1: Grafico a barre delle intensità ordinate
        plt.subplot(1, 2, 1)
        bars = plt.bar(
            [side_labels[i] for i in sorted_idx],
            [side_intensities[i] for i in sorted_idx],
            color=[
                "red" if side_assoc[i] == "background" else "green" for i in sorted_idx
            ],
        )
        plt.axhline(
            y=np.mean(side_intensities),
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Media: {np.mean(side_intensities):.1f}",
        )
        plt.title("Intensità Media Pixel sui Lati (ordinati)")
        plt.ylabel("Intensità media (0-255)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        for i, (bar, idx) in enumerate(zip(bars, sorted_idx)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{side_intensities[idx]:.1f}",
                ha="center",
                va="bottom",
            )
        # Subplot 2: Visualizzazione del contorno con inclinazione
        plt.subplot(1, 2, 2)
        closed_points = np.vstack([points, points[0]])
        plt.plot(
            closed_points[:, 0], closed_points[:, 1], "b-o", linewidth=2, markersize=8
        )
        for i, point in enumerate(points):
            plt.annotate(
                f"P{i + 1}",
                (point[0], point[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            mid_point = (points[i] + points[next_i]) / 2
            plt.annotate(
                f"{side_intensities[i]:.1f}\n{side_inclinations[i]:.1f}°\n{side_assoc[i]}",
                mid_point,
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="red",
                fontweight="bold",
            )
        plt.title("Contorno: Intensità, Inclinazione, Associazione")
        plt.xlabel("X (pixel)")
        plt.ylabel("Y (pixel)")
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        print("\n=== STATISTICHE INTENSITÀ LATI ===")
        print(f"Numero di lati: {len(side_intensities)}")
        print(f"Intensità media: {np.mean(side_intensities):.1f}")
        print(f"Intensità minima: {min(side_intensities):.1f}")
        print(f"Intensità massima: {max(side_intensities):.1f}")
        print(f"Deviazione standard: {np.std(side_intensities):.1f}")
    for i, intensity in enumerate(side_intensities):
        print(f"  {side_labels[i]}: {intensity:.1f}")
    print(f"Inclinazione assoluta media pesata: {mean_weighted_inclination:.2f}°")
    mean = np.average(side_inclinations, weights=weights)
    variance = np.average((np.array(side_inclinations) - mean) ** 2, weights=weights)
    std_weighted_inclination = np.sqrt(variance)
    print(f"Deviazione standard pesata inclinazione: {std_weighted_inclination:.2f}°")
    # trying removing outliers and average again
    filtered_values = np.array(
        [
            [incl, assoc]
            for incl, assoc in zip(side_inclinations, side_assoc)
            if np.abs(np.abs(incl) - np.abs(std_weighted_inclination))
            < std_weighted_inclination
        ]
    ).T
    if filtered_values.size > 0:
        filtered_inclinations, filtered_side_assoc = filtered_values
        print("all angles:", side_inclinations)
        print("non-outliers angles:", filtered_inclinations)
        if len(filtered_inclinations) > 1:
            weights = [dict_weights[assoc] for assoc in filtered_side_assoc]
            filtered_inclinations = np.array(filtered_inclinations, dtype=float)
            # new mean_weighted_inclination = np.average(filtered_inclinations, weights=filtered_side_assoc)
            mean_weighted_inclination = np.average(
                filtered_inclinations, weights=weights
            )
            mean = np.average(filtered_inclinations, weights=weights)
            variance = np.average(
                (np.array(filtered_inclinations) - mean) ** 2, weights=weights
            )
            std_weighted_inclination = np.sqrt(variance)
            print(
                f"Nuova inclinazione assoluta media pesata: {mean_weighted_inclination:.2f}°"
            )
            print(
                f"Nuova deviazione standard pesata inclinazione: {std_weighted_inclination:.2f}°"
            )

    return (
        mean_weighted_inclination,
        variance,
        side_angles,
        side_inclinations,
        side_assoc,
    )


def find_page_contour(thresh, show_step_by_step=False, original_image=None):
    """
    Detect the largest contour that resembles a page.

    Parameters:
        thresh (numpy.ndarray): A binary thresholded image where the contours will be detected.
        show_step_by_step (bool): If True, intermediate images will be displayed for debugging purposes.
        original_image (numpy.ndarray): Original colored image for overlay visualization.

    Returns:
        numpy.ndarray: An approximated polygonal contour of the detected page-like shape.

    Raises:
        ValueError: If no page-like contour is found in the image.

    Steps:
        1. Dilate the thresholded image to close small gaps in the contours.
        2. Detect all external contours in the dilated image.
        3. Sort the contours by area in descending order.
        4. Approximate each contour to reduce the number of points.
        5. Check if the approximated contour has at least four vertices, indicating a page-like shape.
        6. If a suitable contour is found, return it. Otherwise, raise an error.
    """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    if show_step_by_step:
        show_image(dilated, "Dilated")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    angle = None
    approx = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 4:
            rect = cv2.minAreaRect(contour)
            minrect_box = cv2.boxPoints(rect)
            minrect_box = np.int0(minrect_box)
            if show_step_by_step:
                if original_image is not None:
                    overlay = original_image.copy()
                    # Draw thick white border for approx
                    cv2.drawContours(overlay, [approx], -1, (255, 255, 255), 15)
                    # Draw thinner green for approx
                    cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 8)
                    # Draw minAreaRect (red)
                    cv2.drawContours(overlay, [minrect_box], -1, (0, 0, 255), 4)
                    # Draw rectified (yellow)
                    show_image(overlay, "Approx (green), MinAreaRect (red)")
                else:
                    temp_image = np.zeros_like(thresh)
                    cv2.drawContours(temp_image, [approx], -1, (255, 255, 255), 3)
                    show_image(temp_image, "Detected Contour")

            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            angle, d_angle, _, _, _ = contour_side_intensity(
                approx, gray, show_plot=show_step_by_step
            )
            if np.abs(angle) < np.abs(d_angle):  # noise
                angle = 0

            # Return the rectified rectangle as the best rectangle
            return approx, angle

    # If no suitable contour is found, return None
    return None, None
