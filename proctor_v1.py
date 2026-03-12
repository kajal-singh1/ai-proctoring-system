import cv2
import time
from violation_logger import init_logger, log_violation

# ── Load face detection model ──────────────────────────────────────
MODEL_FILE  = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
print("✅ Model loaded.")

# ── Initialize logger ──────────────────────────────────────────────
init_logger()

# ── Open webcam ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# ── Violation cooldown settings ────────────────────────────────────
# We don't want to log 30 violations per second — only log if the
# violation persists for ALERT_SECONDS and then cooldown for COOLDOWN_SECONDS
ALERT_SECONDS    = 2   # face must be missing for 2s before logging
COOLDOWN_SECONDS = 5   # wait 5s before logging the same violation again

# Track when each violation type was last logged
last_logged = {
    "NO_FACE":        0,
    "MULTIPLE_FACES": 0
}

# Track how long a violation has been continuously happening
violation_start = {
    "NO_FACE":        None,
    "MULTIPLE_FACES": None
}

print("🎥 Proctoring session started. Press Q to quit.")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    now = time.time()

    # ── Run face detection ─────────────────────────────────────────
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Determine current violation state ─────────────────────────
    if face_count == 0:
        current_violation = "NO_FACE"
    elif face_count >= 2:
        current_violation = "MULTIPLE_FACES"
    else:
        current_violation = None

    # ── Violation timing + cooldown logic ─────────────────────────
    for v_type in ["NO_FACE", "MULTIPLE_FACES"]:
        if current_violation == v_type:
            # Start the timer if this is a fresh violation
            if violation_start[v_type] is None:
                violation_start[v_type] = now

            # How long has this violation been happening?
            duration = now - violation_start[v_type]
            time_since_last_log = now - last_logged[v_type]

            # Log only if persisted long enough AND cooldown has passed
            if duration >= ALERT_SECONDS and time_since_last_log >= COOLDOWN_SECONDS:
                details = f"face_count={face_count}"
                log_violation(v_type, details)
                last_logged[v_type] = now
        else:
            # Reset the start timer when violation clears
            violation_start[v_type] = None

    # ── Build status display ───────────────────────────────────────
    if current_violation == "NO_FACE":
        status_text  = "⚠ NO FACE DETECTED"
        status_color = (0, 0, 255)    # Red
    elif current_violation == "MULTIPLE_FACES":
        status_text  = "⚠ MULTIPLE FACES"
        status_color = (0, 165, 255)  # Orange
    else:
        status_text  = "✓ OK — 1 Face"
        status_color = (0, 255, 0)    # Green

    # ── Draw status bar at top of frame ───────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)  # black bar
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # ── Show face count bottom-left ────────────────────────────────
    cv2.putText(frame, f"Faces: {face_count}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Proctor v1", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Session ended. Violations saved to violations.csv")