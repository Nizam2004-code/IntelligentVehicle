import cv2
import time
import math
import threading
import winsound
from ultralytics import YOLO

# ============================================================
# ROAD MONITORING CONFIGURATION
# ============================================================

MODEL_NAME = "yolo11n.pt"
CAMERA_INDEX = 0
CONFIDENCE = 0.40

# Approximate distance settings
FOCAL_LENGTH = 700

KNOWN_WIDTH = {
    "person":0.5,
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.5,
    "motorcycle": 0.8,
    "bicycle": 0.6,
}

# Distance thresholds
WARNING_DISTANCE = 12.0
CRITICAL_DISTANCE = 6.0

# Objects we want to detect
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# BUZZER
# ============================================================

BUZZER_ENABLED = True

buzzer_state = "SAFE"
buzzer_lock = threading.Lock()


def buzzer_loop(state):

    while True:

        with buzzer_lock:
            if buzzer_state != state:
                return

        try:

            if state == "CRITICAL":

                winsound.Beep(1100, 200)
                time.sleep(0.15)

                winsound.Beep(1100, 200)
                time.sleep(0.20)

            elif state == "WARNING":

                winsound.Beep(750, 150)
                time.sleep(0.9)

            else:
                return

        except Exception:
            return


def set_buzzer(state):

    global buzzer_state

    with buzzer_lock:

        if buzzer_state == state:
            return

        buzzer_state = state

    if BUZZER_ENABLED and state in ["WARNING", "CRITICAL"]:

        threading.Thread(
            target=buzzer_loop,
            args=(state,),
            daemon=True
        ).start()


# ============================================================
# DISTANCE ESTIMATION
# ============================================================

def estimate_distance(object_name, width_pixels):

    if object_name not in KNOWN_WIDTH:
        return None

    if width_pixels <= 0:
        return None

    real_width = KNOWN_WIDTH[object_name]

    distance = (
        FOCAL_LENGTH * real_width
    ) / width_pixels

    return distance


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_NAME)

print("YOLO model loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("Road monitoring started.")
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

previous_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break


    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    nearest_distance = None
    nearest_object = None

    object_count = 0


    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])


            if class_id not in TARGET_CLASSES:
                continue


            object_name = TARGET_CLASSES[class_id]

            object_count += 1


            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            box_width = x2 - x1


            # ------------------------------------------------
            # DISTANCE
            # ------------------------------------------------

            distance = estimate_distance(
                object_name,
                box_width
            )


            # ------------------------------------------------
            # CHECK IF OBJECT IS IN FRONT CENTER
            # ------------------------------------------------

            object_center = (x1 + x2) / 2

            frame_center = width / 2

            center_difference = abs(
                object_center - frame_center
            )

            center_zone = width * 0.25

            in_front = (
                center_difference <= center_zone
            )


            # ------------------------------------------------
            # FIND NEAREST OBJECT
            # ------------------------------------------------

            if (
                distance is not None
                and in_front
            ):

                if (
                    nearest_distance is None
                    or distance < nearest_distance
                ):

                    nearest_distance = distance
                    nearest_object = object_name


            # ------------------------------------------------
            # BOX COLOR
            # ------------------------------------------------

            box_color = (255, 255, 0)

            if distance is not None and in_front:

                if distance <= CRITICAL_DISTANCE:

                    box_color = (0, 0, 255)

                elif distance <= WARNING_DISTANCE:

                    box_color = (0, 165, 255)

                else:

                    box_color = (0, 255, 0)


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            if distance is not None:

                label = (
                    f"{object_name} "
                    f"{confidence:.0%} "
                    f"{distance:.1f}m"
                )

            else:

                label = (
                    f"{object_name} "
                    f"{confidence:.0%}"
                )


            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


    # ========================================================
    # ROAD SAFETY STATUS
    # ========================================================

    road_status = "SAFE"
    warning_message = "ROAD CLEAR"


    if nearest_distance is not None:

        if nearest_distance <= CRITICAL_DISTANCE:

            road_status = "CRITICAL"

            warning_message = (
                f"VERY CLOSE: "
                f"{nearest_object.upper()} "
                f"{nearest_distance:.1f} m"
            )


        elif nearest_distance <= WARNING_DISTANCE:

            road_status = "WARNING"

            warning_message = (
                f"CAUTION: "
                f"{nearest_object.upper()} "
                f"{nearest_distance:.1f} m"
            )


    # ========================================================
    # BUZZER
    # ========================================================

    set_buzzer(road_status)


    # ========================================================
    # STATUS BANNER
    # ========================================================

    if road_status == "CRITICAL":

        banner_color = (0, 0, 255)

    elif road_status == "WARNING":

        banner_color = (0, 165, 255)

    else:

        banner_color = (0, 150, 0)


    cv2.rectangle(
        frame,
        (10, 10),
        (720, 65),
        banner_color,
        -1
    )


    cv2.putText(
        frame,
        f"{road_status} | {warning_message}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Objects detected: {object_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    if nearest_distance is not None:

        cv2.putText(
            frame,
            f"Nearest: {nearest_object} "
            f"{nearest_distance:.1f} m",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / (
        current_time - previous_time
    )

    previous_time = current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CENTER ROAD ZONE
    # ========================================================

    left_zone = int(width * 0.25)
    right_zone = int(width * 0.75)


    cv2.line(
        frame,
        (left_zone, 0),
        (left_zone, height),
        (255, 255, 0),
        1
    )

    cv2.line(
        frame,
        (right_zone, 0),
        (right_zone, height),
        (255, 255, 0),
        1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Heavy Vehicle Road Monitoring",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

with buzzer_lock:
    buzzer_state = "SAFE"

cap.release()

cv2.destroyAllWindows()

print("Road monitoring stopped.")