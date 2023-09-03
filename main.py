import cv2
import imutils
import numpy as np


# Funktion zur Berechnung des Orangenanteils in einem Bild
def calculate_orange_percentage(image):
    # Orangenbereich in HSV-Farbraum definieren
    lower_orange = np.array([113, 35, 29])
    upper_orange = np.array([172, 210, 215])

    # Bild in HSV umwandeln
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Maske für den Orangenbereich erstellen
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Summe der weißen Pixel in der Maske berechnen (Orangenanteil)
    orange_pixels = cv2.countNonZero(orange_mask)

    # Gesamtzahl der Pixel im Bild berechnen
    total_pixels = image.shape[0] * image.shape[1]

    # Orangenanteil in Prozent berechnen
    orange_percentage = (orange_pixels / total_pixels) * 100

    return orange_percentage


# Kamera initialisieren
cap = cv2.VideoCapture(1)
prev_max_orange_box = None
while True:
    ret, frame = cap.read()  # Einzelbild erfassen

    # Bild in 30x40 Kästchen unterteilen
    height, width, _ = frame.shape
    box_height = height // 30
    box_width = width // 40

    max_orange_box = None
    max_orange_percentage = 0

    for row in range(30):
        for col in range(40):
            x1 = col * box_width
            x2 = x1 + box_width
            y1 = row * box_height
            y2 = y1 + box_height

            box = frame[y1:y2, x1:x2]
            orange_percentage = calculate_orange_percentage(box)

            if orange_percentage > max_orange_percentage:
                max_orange_percentage = orange_percentage
                max_orange_box = (row, col)

    # Kästchen mit dem höchsten Orangenanteil markieren
    if max_orange_box is not None:
        row, col = max_orange_box
        x1 = col * box_width
        x2 = x1 + box_width
        y1 = row * box_height
        y2 = y1 + box_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if max_orange_box != prev_max_orange_box:
            print(f"Markiertes Kästchen: Reihe={row}, Spalte={col}")
            prev_max_orange_box = max_orange_box

    # Live-Bild anzeigen
    cv2.imshow("Live Bild", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()