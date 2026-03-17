import cv2
import mediapipe as mp
import numpy as np
from violation_logger import init_logger, log_violation
import time
from collections import deque

# ── Initialize MediaPipe ───────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

print("Gaze tracker loaded.")

# ── Landmark indices ───────────────────────────────────────────────
# Eye corners
LEFT_EYE_LEFT = 33      # leftmost point of left eye
LEFT_EYE_RIGHT  = 133   # rightmost point of left eye
RIGHT_EYE_LEFT  = 362   # leftmost point of right eye
RIGHT_EYE_RIGHT = 263   # rightmost point of right eye

# Iris centers (only available with refine_landmarks=True)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# Gaze thresholds
GAZE_LEFT_THRESHOLD  = 0.42
GAZE_RIGHT_THRESHOLD = 0.54

# Smoothing buffer — average last 10 frames
ratio_buffer = deque(maxlen=10)

# Violation cooldown
ALERT_SECONDS    = 1.5
COOLDOWN_SECONDS = 5

def get_coords(landmark, w, h):
    """Convert normalized coords to pixel coords."""
    return int(landmark.x * w), int(landmark.y * h)

def compute_iris_ratio(eye_left, eye_right, iris_center):
    """
    Compute where iris sits between eye corners.
    Returns value 0-1. Center ~0.5, left gaze <0.35, right gaze >0.65
    """

    eye_width = eye_right[0] - eye_left[0]
    if eye_width == 0:
        return 0.5  # avoid division by zero
    ratio = (iris_center[0] - eye_left[0])/ eye_width
    return ratio

def get_gaze_direction(left_ratio, right_ratio):
    avg_ratio = (left_ratio + right_ratio) / 2.0
    if avg_ratio > GAZE_RIGHT_THRESHOLD:
        return "LOOKING_LEFT", avg_ratio
    elif avg_ratio < GAZE_LEFT_THRESHOLD:
        return "LOOKING_RIGHT", avg_ratio
    else:
        return "CENTER", avg_ratio
    
# ── Initialize logger ──────────────────────────────────────────────
init_logger()

# ── Violation timing ───────────────────────────────────────────────
last_logged = {"GAZE_LEFT": 0, "GAZE_RIGHT": 0}
violation_start = {"GAZE_LEFT": None, "GAZE_RIGHT": None}

# ── Open webcam ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Gaze tracker running. Press Q to quit.")
print("Try looking LEFT and RIGHT to test detection")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h,w = frame.shape[:2]
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    gaze_direction = "NO_FACE"
    avg_ratio = 0.5
    left_ratio = 0.5
    right_ratio = 0.5

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # ── Get eye corner pixel coords ────────────────────────────
        ll = get_coords(lm[LEFT_EYE_LEFT],    w, h)
        lr = get_coords(lm[LEFT_EYE_RIGHT],   w, h)
        rl = get_coords(lm[RIGHT_EYE_LEFT],   w, h)
        rr = get_coords(lm[RIGHT_EYE_RIGHT],  w, h)

        # ── Get iris center pixel coords ───────────────────────────
        li = get_coords(lm[LEFT_IRIS_CENTER],  w, h)
        ri = get_coords(lm[RIGHT_IRIS_CENTER], w, h)

        # ── Compute ratios ─────────────────────────────────────────
        left_ratio  = compute_iris_ratio(ll, lr, li)
        right_ratio = compute_iris_ratio(rl, rr, ri)

        # Add current ratio to buffer
        raw_ratio = (left_ratio + right_ratio) / 2.0
        ratio_buffer.append(raw_ratio)

        # Use smoothed average instead of raw single frame
        smoothed_ratio = sum(ratio_buffer) / len(ratio_buffer)
        gaze_direction, avg_ratio = get_gaze_direction(smoothed_ratio, smoothed_ratio)

        # ── Draw eye corners ───────────────────────────────────────
        for pt in [ll, lr, rl, rr]:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        # ── Draw iris centers ──────────────────────────────────────
        cv2.circle(frame, li, 4, (0, 255, 255), -1)
        cv2.circle(frame, ri, 4, (0, 255, 255), -1)

        # ── Draw gaze ratio bar ────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = 20, 80, 200, 20
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill = int(avg_ratio * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), (0, 255, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        # Threshold markers on the bar
        cv2.line(frame,
                 (bar_x + int(GAZE_LEFT_THRESHOLD * bar_w), bar_y),
                 (bar_x + int(GAZE_LEFT_THRESHOLD * bar_w), bar_y + bar_h),
                 (0, 0, 255), 2)
        cv2.line(frame,
                 (bar_x + int(GAZE_RIGHT_THRESHOLD * bar_w), bar_y),
                 (bar_x + int(GAZE_RIGHT_THRESHOLD * bar_w), bar_y + bar_h),
                 (0, 0, 255), 2)

        cv2.putText(frame, f"Ratio: {avg_ratio:.2f}", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Violation logic ────────────────────────────────────────────
    v_map = {
        "LOOKING_LEFT":  "GAZE_LEFT",
        "LOOKING_RIGHT": "GAZE_RIGHT"
    }
    current_violation = v_map.get(gaze_direction, None)

    for v_type in ["GAZE_LEFT", "GAZE_RIGHT"]:
        if current_violation == v_type:
            if violation_start[v_type] is None:
                violation_start[v_type] = now
            duration = now - violation_start[v_type]
            time_since_last = now - last_logged[v_type]
            if duration >= ALERT_SECONDS and time_since_last >= COOLDOWN_SECONDS:
                log_violation(v_type, f"ratio={avg_ratio:.2f}")
                last_logged[v_type] = now
        else:
            violation_start[v_type] = None

    # ── Status display ─────────────────────────────────────────────
    if gaze_direction == "CENTER":
        status_text  = "✓ Gaze: CENTER"
        status_color = (0, 255, 0)
    elif gaze_direction == "LOOKING_LEFT":
        status_text  = "⚠ LOOKING LEFT"
        status_color = (0, 0, 255)
    elif gaze_direction == "LOOKING_RIGHT":
        status_text  = "⚠ LOOKING RIGHT"
        status_color = (0, 165, 255)
    else:
        status_text  = "NO FACE"
        status_color = (128, 128, 128)

    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Gaze Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Gaze tracking session ended.")