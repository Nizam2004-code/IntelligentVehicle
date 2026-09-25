import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import requests
import threading
import winsound

# ============================================================
# INTELLIGENT HEAVY-VEHICLE DRIVER MONITORING
# MediaPipe Tasks API - compatible with current MediaPipe 1.x
# ============================================================

CAMERA_INDEX = 0

# Face Landmarker model
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Detection thresholds
EYE_BLINK_THRESHOLD = 0.55       # blendshape eyeBlink score
EYE_CLOSED_TIME = 1.5            # continuous seconds

YAWN_JAW_THRESHOLD = 0.55        # jawOpen blendshape score
YAWN_TIME = 1.0

DISTRACTION_TIME = 2.7
HEAD_YAW_THRESHOLD = 0.18        # normalized head offset

# Safety Engine integration
# Keep False while testing this program alone.
SEND_TO_SERVER = False
SERVER_URL = "http://127.0.0.1:5000"

# ============================================================
# LANDMARK INDICES
# ============================================================

# Used as a fallback for EAR/MAR if blendshapes are unavailable.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454


# ============================================================
# BUZZER / WARNING CONFIGURATION
# ============================================================

# Software buzzer using the Windows laptop speaker.
# This simulates the cabin warning buzzer for the hackathon PoC.
BUZZER_ENABLED = True

WARNING_FREQ = 750
WARNING_DURATION_MS = 140

CRITICAL_FREQ = 1100
CRITICAL_DURATION_MS = 220

_buzzer_state = "SAFE"
_buzzer_thread = None
_buzzer_lock = threading.Lock()


def _beep_loop(state):
    """Play a non-blocking warning pattern for the current safety state."""
    while True:
        with _buzzer_lock:
            if _buzzer_state != state or not BUZZER_ENABLED:
                return

        try:
            if state == "CRITICAL":
                winsound.Beep(CRITICAL_FREQ, CRITICAL_DURATION_MS)
                time.sleep(0.18)
                winsound.Beep(CRITICAL_FREQ, CRITICAL_DURATION_MS)
                time.sleep(0.25)

            elif state == "WARNING":
                winsound.Beep(WARNING_FREQ, WARNING_DURATION_MS)
                time.sleep(0.85)

            else:
                return

        except Exception:
            # Never allow the buzzer to stop the AI camera system.
            return


def set_buzzer_state(state):
    """Start/stop the appropriate buzzer pattern without blocking the camera."""
    global _buzzer_state, _buzzer_thread

    if not BUZZER_ENABLED:
        return

    with _buzzer_lock:
        if _buzzer_state == state:
            return

        _buzzer_state = state

    if state in ("WARNING", "CRITICAL"):
        _buzzer_thread = threading.Thread(
            target=_beep_loop,
            args=(state,),
            daemon=True
        )
        _buzzer_thread.start()


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return

    print("Face Landmarker model not found.")
    print("Downloading model from the official MediaPipe model repository...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Face Landmarker model downloaded successfully.")
    except Exception as exc:
        raise RuntimeError(
            "Could not download face_landmarker.task. "
            "Check your internet connection and try again."
        ) from exc


# ============================================================
# MATH HELPERS
# ============================================================

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((landmark.x * width, landmark.y * height))

    p1, p2, p3, p4, p5, p6 = points

    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, width, height):
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]

    horizontal = distance(
        (left.x * width, left.y * height),
        (right.x * width, right.y * height)
    )

    vertical = distance(
        (top.x * width, top.y * height),
        (bottom.x * width, bottom.y * height)
    )

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def get_blendshape_scores(result):
    """
    Return useful Face Landmarker blendshape scores.
    MediaPipe returns a list of Category objects.
    """
    scores = {}

    if not result.face_blendshapes:
        return scores

    for category in result.face_blendshapes[0]:
        name = category.category_name
        scores[name] = float(category.score)

    return scores


def send_driver_data(data):
    if not SEND_TO_SERVER:
        return

    try:
        requests.post(
            f"{SERVER_URL}/driver",
            json=data,
            timeout=0.15
        )
    except requests.RequestException:
        # Keep camera monitoring alive if the server is unavailable.
        pass


# ============================================================
# SETUP
# ============================================================

ensure_model()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check Windows camera permissions "
        "or change CAMERA_INDEX."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Timers
eye_closed_start = None
yawn_start = None
distraction_start = None

# Event counters
drowsiness_events = 0
yawn_events = 0
distraction_events = 0

previous_drowsiness = False
previous_yawning = False
previous_distraction = False

last_payload_time = 0.0
payload_interval = 0.20

timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read frame from webcam.")
            break

        # Mirror the driver's camera view.
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # MediaPipe Tasks expects an RGB image.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Default state
        face_detected = bool(result.face_landmarks)

        drowsiness = False
        yawning = False
        distraction = False

        head_direction = "CENTER"

        eye_blink_left = 0.0
        eye_blink_right = 0.0
        jaw_open = 0.0

        ear = 0.0
        mar = 0.0

        # ----------------------------------------------------
        # FACE DETECTED
        # ----------------------------------------------------

        if face_detected:

            face = result.face_landmarks[0]

            # -----------------------------------------------
            # 1. BLINK / EYE CLOSURE
            # -----------------------------------------------

            blendshapes = get_blendshape_scores(result)

            eye_blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
            eye_blink_right = blendshapes.get("eyeBlinkRight", 0.0)

            # Average blink probability.
            eye_blink_score = (
                eye_blink_left + eye_blink_right
            ) / 2.0

            # Fallback EAR for visibility/debugging.
            ear_left = eye_aspect_ratio(
                face, LEFT_EYE, width, height
            )
            ear_right = eye_aspect_ratio(
                face, RIGHT_EYE, width, height
            )
            ear = (ear_left + ear_right) / 2.0

            eyes_closed = eye_blink_score >= EYE_BLINK_THRESHOLD

            if eyes_closed:

                if eye_closed_start is None:
                    eye_closed_start = time.time()

                eye_closed_duration = (
                    time.time() - eye_closed_start
                )

                if eye_closed_duration >= EYE_CLOSED_TIME:
                    drowsiness = True

            else:
                eye_closed_start = None

            # -----------------------------------------------
            # 2. YAWNING
            # -----------------------------------------------

            jaw_open = blendshapes.get("jawOpen", 0.0)

            # Fallback mouth ratio for debugging.
            mar = mouth_aspect_ratio(
                face, width, height
            )

            yawning_now = jaw_open >= YAWN_JAW_THRESHOLD

            if yawning_now:

                if yawn_start is None:
                    yawn_start = time.time()

                yawn_duration = time.time() - yawn_start

                if yawn_duration >= YAWN_TIME:
                    yawning = True

            else:
                yawn_start = None

            # -----------------------------------------------
            # 3. HEAD ORIENTATION / DISTRACTION
            # -----------------------------------------------

            nose_x = face[NOSE_TIP].x
            left_x = face[FACE_LEFT].x
            right_x = face[FACE_RIGHT].x

            face_width = right_x - left_x

            if face_width > 0:

                face_center_x = (left_x + right_x) / 2.0

                # Positive = right side of image.
                head_offset = (
                    nose_x - face_center_x
                ) / face_width

                if head_offset < -HEAD_YAW_THRESHOLD:
                    head_direction = "LEFT"

                elif head_offset > HEAD_YAW_THRESHOLD:
                    head_direction = "RIGHT"

                else:
                    head_direction = "CENTER"

                looking_away = head_direction != "CENTER"

                if looking_away:

                    if distraction_start is None:
                        distraction_start = time.time()

                    distraction_duration = (
                        time.time() - distraction_start
                    )

                    if distraction_duration >= DISTRACTION_TIME:
                        distraction = True

                else:
                    distraction_start = None

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        else:
            eye_closed_start = None
            yawn_start = None
            distraction_start = None

        # ----------------------------------------------------
        # EVENT COUNTERS
        # ----------------------------------------------------

        if drowsiness and not previous_drowsiness:
            drowsiness_events += 1

        if yawning and not previous_yawning:
            yawn_events += 1

        if distraction and not previous_distraction:
            distraction_events += 1

        previous_drowsiness = drowsiness
        previous_yawning = yawning
        previous_distraction = distraction

        # ----------------------------------------------------
        # WARNING / RISK LEVEL
        # ----------------------------------------------------

        # Driver-only safety level for the prototype.
        # Drowsiness is treated as CRITICAL.
        # Distraction, yawning and missing face are WARNING.
        if drowsiness:
            warning_level = "CRITICAL"
            warning_message = "DROWSINESS ALERT - TAKE A BREAK"

        elif distraction:
            warning_level = "WARNING"
            warning_message = "DISTRACTION ALERT - KEEP EYES ON ROAD"

        elif yawning:
            warning_level = "WARNING"
            warning_message = "FATIGUE WARNING - YAWNING DETECTED"

        elif not face_detected:
            warning_level = "WARNING"
            warning_message = "DRIVER NOT DETECTED"

        else:
            warning_level = "SAFE"
            warning_message = "DRIVER ATTENTIVE"

        # Start the corresponding software buzzer pattern.
        set_buzzer_state(warning_level)

        # ----------------------------------------------------
        # DRIVER STATUS
        # ----------------------------------------------------

        if drowsiness:
            driver_status = "DROWSINESS DETECTED"

        elif distraction:
            driver_status = "DISTRACTION DETECTED"

        elif yawning:
            driver_status = "YAWNING DETECTED"

        elif not face_detected:
            driver_status = "NO DRIVER FACE"

        else:
            driver_status = "ATTENTIVE"

        # ----------------------------------------------------
        # PROTOTYPE FATIGUE SCORE
        # ----------------------------------------------------

        fatigue_score = 0

        if drowsiness:
            fatigue_score += 60

        if distraction:
            fatigue_score += 30

        if yawning:
            fatigue_score += 15

        if not face_detected:
            fatigue_score += 20

        fatigue_score = min(fatigue_score, 100)

        # ----------------------------------------------------
        # UI PANEL
        # ----------------------------------------------------

        if drowsiness or distraction:
            status_color = (0, 0, 255)

        elif yawning or not face_detected:
            status_color = (0, 165, 255)

        else:
            status_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (10, 10),
            (620, 345),
            (20, 20, 20),
            -1
        )

        # Large warning banner.
        if warning_level == "CRITICAL":
            banner_color = (0, 0, 255)
        elif warning_level == "WARNING":
            banner_color = (0, 165, 255)
        else:
            banner_color = (0, 180, 0)

        cv2.rectangle(
            frame,
            (10, 10),
            (620, 58),
            banner_color,
            -1
        )

        cv2.putText(
            frame,
            f"{warning_level}: {warning_message}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATUS: {driver_status}",
            (25, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"Eye Blink: {((eye_blink_left + eye_blink_right) / 2):.2f}",
            (25, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}  Jaw Open: {jaw_open:.2f}",
            (25, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Mouth Ratio: {mar:.2f}",
            (25, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"HEAD: {head_direction}",
            (25, 196),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Fatigue Score: {fatigue_score}/100",
            (25, 222),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Drowsiness Events: {drowsiness_events}",
            (25, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Yawning Events: {yawn_events}",
            (25, 276),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Distraction Events: {distraction_events}",
            (25, 302),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (width - 100, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ----------------------------------------------------
        # SAFETY ENGINE PAYLOAD
        # ----------------------------------------------------

        payload = {
            "face_detected": face_detected,
            "drowsiness": drowsiness,
            "distraction": distraction,
            "yawning": yawning,
            "driver_status": driver_status,
            "head_direction": head_direction,
            "eye_aspect_ratio": round(float(ear), 3),
            "mouth_aspect_ratio": round(float(mar), 3),
            "eye_blink_score": round(
                float((eye_blink_left + eye_blink_right) / 2),
                3
            ),
            "jaw_open_score": round(float(jaw_open), 3),
            "fatigue_score": fatigue_score,
            "warning_level": warning_level,
            "warning_message": warning_message,
            "buzzer": warning_level != "SAFE",
            "drowsiness_events": drowsiness_events,
            "distraction_events": distraction_events,
            "yawn_events": yawn_events,
            "timestamp": time.time()
        }

        if time.time() - last_payload_time >= payload_interval:
            send_driver_data(payload)
            last_payload_time = time.time()

        cv2.imshow(
            "Driver Monitoring AI - MediaPipe Tasks",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


with _buzzer_lock:
    _buzzer_state = "SAFE"

cap.release()
cv2.destroyAllWindows()
