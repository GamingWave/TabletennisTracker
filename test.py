import time
import cv2
import numpy as np
from enum import Enum
import threading
from stopwatch import Stopwatch


class SpielerTypes(Enum):
    Rot = "red"
    Blau = "blue"

stopwatch = Stopwatch()
stopwatch.reset()

im_ballwechsel = None
blau = 0
rot = 0
angabe = None
angabe_spieler = None
angabenummer = 0
angabe_thread = False
letzter_sprung = None
jump_time = 0
counting = False
"""
def watch():
    global rot, blau, im_ballwechsel, counting
    stopwatch.restart()

    while True:
        time.sleep(0.2)
        if stopwatch.duration > 2.5:
            im_ballwechsel = False
            if counting != True:
                counting = True
            break
"""

class WatchThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_flag = threading.Event()

    def run(self):
        global rot, blau, im_ballwechsel, counting, stopallthread, jump_time

        while not self.stop_flag.is_set():
            stopwatch.restart()
            ball = jump_time
            while True:
                if jump_time != ball:
                    self.stop()
                    break

                time.sleep(0.2)
                if stopwatch.duration > 2.5:
                    im_ballwechsel = False
                    if counting != True:
                        counting = True
                    break



    def stop(self):
        self.stop_flag.set()

stopallthread = False
class AngabeTimer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_flag = threading.Event()

    def run(self):
        global angabe, angabe_thread, letzter_sprung
        while not self.stop_flag.is_set():
            time.sleep(2)
            if row < 10:
                print("Angabe Frei")
                angabe = True
                letzter_sprung = None
                angabe_thread = False
                self.stop()
    def stop(self):
        self.stop_flag.set()


def sprung(spieler):
    global angabe, im_ballwechsel, jump_time, letzter_sprung, stopallthread, blau, rot
    jump_time = jump_time + 1
    if im_ballwechsel == True or angabe == True:
        angabe = False
        im_ballwechsel = True

        print(jump_time)
        if(spieler == "Blau"):
            if letzter_sprung == "Rot" or letzter_sprung is None:

                stopallthread = True
                thread = WatchThread()
                stopallthread = False
                thread.start()


            if letzter_sprung == "Blau":
                rot = rot + 1
                print(f"Punkt für Rot\nNeuer Spielstand: {blau}:{rot}")
                im_ballwechsel = False
            letzter_sprung = "Blau"

        if spieler == "Rot":
            if letzter_sprung == "Blau" or letzter_sprung is None:

                stopallthread = True
                thread = WatchThread()
                stopallthread = False
                thread.start()

            if letzter_sprung == "Rot":
                blau = blau + 1
                print(f"Punkt für Blau\nNeuer Spielstand: {blau}:{rot}")
                im_ballwechsel = False
            letzter_sprung = "Rot"





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
cap = cv2.VideoCapture(0)
prev_max_orange_box = None
prev_row = None
prev_boxes = []

# Überwachung der Änderung der Kästchenpositionen
is_jumping = False
threshold = 1  # Schwellenwert für Änderung der Reihenposition
angabe = False
angabe_spieler = SpielerTypes.Blau
im_ballwechsel = False
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

        if prev_max_orange_box != max_orange_box:
            # print(f"Markiertes Kästchen: Reihe={row}, Spalte={col}")
            prev_max_orange_box = max_orange_box

        # Überwachung der Änderung der Kästchenpositionen in vertikaler Richtung
        prev_boxes.append((row, col))
        if len(prev_boxes) > 10:
            prev_boxes.pop(0)

        if len(prev_boxes) == 10:
            # Überprüfen, ob sich die Reihenpositionen in den letzten 10 Frames geändert haben
            row_changes = [prev_boxes[i][0] - prev_boxes[i - 1][0] for i in range(1, 10)]
            if any(change > threshold for change in row_changes) and not is_jumping:
                is_jumping = True
                if col > 21:
                    sprung("Rot")
                else:
                    sprung("Blau")
            elif all(change <= threshold for change in row_changes) and is_jumping:
                is_jumping = False



    if im_ballwechsel == False:
        if row < 10:
            if angabe == False:
                if angabe_thread == False:
                    stop = AngabeTimer()
                    stop.start()
                    angabe_thread = True
        if counting == True:
            if letzter_sprung == "Blau":
                rot = rot + 1
                print(f"Punkt für Rot\nNeuer Spielstand: {blau}:{rot}")
                counting = False
            if letzter_sprung == "Rot":
                blau = blau + 1
                print(f"Punkt für Blau\nNeuer Spielstand: {blau}:{rot}")
                counting = False

    frame = cv2.line(frame, (20, 240), (320, 240), (255, 0, 0), 1)
    frame = cv2.line(frame, (320, 240), (620, 240), (0, 0, 255), 1)
    frame = cv2.line(frame, (320, 240), (320, 200), (0, 255, 0), 1)
    # Live-Bild anzeigen
    cv2.imshow("Live Bild", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
