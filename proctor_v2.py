import cv2
import mediapipe as mp
import numpy as np
from violation_logger import init_logger, log_violation
import time
from collections import deque

# ═══════════════════════════════════════════════════════════════════
#  CONFIG — tweak these to adjust sensitivity
# ═══════════════════════════════════════════════════════════════════
GAZE_LEFT_THRESHOLD  = 0.45
GAZE_RIGHT_THRESHOLD = 0.52
YAW_THRESHOLD        = 25
PITCH_THRESHOLD      = 13
YAW_OFFSET           = 5
ROLL_FLIP_LIMIT      = 90
ALERT_SECONDS        = 1.5
COOLDOWN_SECONDS     = 5

# ═══════════════════════════════════════════════════════════════════
#  MEDIAPIPE SETUP
# ═══════════════════════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 2,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
print("MideaPipe Loaded.")

# ═══════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════════
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

POSE_LANDMARKS = {
    "nose_tip"   : 1,
    "chin"       : 152,
    "left_eye"   : 33,
    "right_eye"  : 263,
    "left_mouth" : 61,
    "right_mouth": 291
}

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    (43.3,   32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    (28.9,  -28.9,  -24.1),
], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════
#  SMOOTHING BUFFERS
# ═══════════════════════════════════════════════════════════════════
ratio_buffer = deque(maxlen = 5)
yaw_buffer = deque(maxlen = 10)
pitch_buffer = deque(maxlen=10)

# ═══════════════════════════════════════════════════════════════════
#  VIOLATION STATE
# ═══════════════════════════════════════════════════════════════════
ALL_VIOLATIONS = [
    "NO_FACE", "MULTIPLE_FACES",
    "GAZE_LEFT", "GAZE_RIGHT",
    "HEAD_TURN", "HEAD_DOWN"
]
last_logged     = {v: 0    for v in ALL_VIOLATIONS}
violation_start = {v: None for v in ALL_VIOLATIONS}

# ═══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def get_coords(landmark, w, h):
    return int(landmark.x * w), int(landmark.y*h)

def compute_iris_ratio(eye_left, eye_right, iris_center):
    eye_width = eye_right[0] - eye_left[0]
    if eye_width == 0:
        return 0.5
    return (iris_center[0] - eye_left[0]) / eye_width

def estimate_head_pose(landmarks, w, h):
    pts = []
    for idx in POSE_LANDMARKS.values():
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))
    image_points = np.array(pts, dtype=np.float64)

    focal      = w
    cam_matrix = np.array([
        [focal, 0, w/2],
        [0, focal, h/2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( rmat[2,1], rmat[2,2]))
        yaw   = np.degrees(np.arctan2(-rmat[2,0], sy))
        roll  = np.degrees(np.arctan2( rmat[1,0], rmat[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1,2], rmat[1,1]))
        yaw   = np.degrees(np.arctan2(-rmat[2,0], sy))
        roll  = 0.0

    if pitch > 90:   pitch -= 180
    elif pitch < -90: pitch += 180
    return yaw, pitch, roll

def check_and_log(v_type, is_active, now, details=""):
    """Unified violation timer + cooldown logic."""
    if is_active:
        if violation_start[v_type] is None:
            violation_start[v_type] = now
        duration        = now - violation_start[v_type]
        time_since_last = now - last_logged[v_type]
        if duration >= ALERT_SECONDS and time_since_last >= COOLDOWN_SECONDS:
            log_violation(v_type, details)
            last_logged[v_type] = now
    else:
        violation_start[v_type] = None

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
init_logger()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("🎓 AI Proctor v2 running. Press Q to quit.")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    now  = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    face_count    = 0
    gaze_status   = "NO_FACE"
    head_status   = "NO_FACE"
    smooth_yaw    = 0.0
    smooth_pitch  = 0.0
    avg_ratio     = 0.5

    if results.multi_face_landmarks:
        face_count = len(results.multi_face_landmarks)

        # ── Use first face for gaze + head pose ───────────────────
        lm = results.multi_face_landmarks[0].landmark

        # ── GAZE ──────────────────────────────────────────────────
        ll = get_coords(lm[LEFT_EYE_LEFT],    w, h)
        lr = get_coords(lm[LEFT_EYE_RIGHT],   w, h)
        rl = get_coords(lm[RIGHT_EYE_LEFT],   w, h)
        rr = get_coords(lm[RIGHT_EYE_RIGHT],  w, h)
        li = get_coords(lm[LEFT_IRIS_CENTER],  w, h)
        ri = get_coords(lm[RIGHT_IRIS_CENTER], w, h)

        left_ratio  = compute_iris_ratio(ll, lr, li)
        right_ratio = compute_iris_ratio(rl, rr, ri)
        raw_ratio   = (left_ratio + right_ratio) / 2.0
        ratio_buffer.append(raw_ratio)
        avg_ratio = sum(ratio_buffer) / len(ratio_buffer)

        if avg_ratio > GAZE_RIGHT_THRESHOLD:
            gaze_status = "LOOKING_LEFT"
        elif avg_ratio < GAZE_LEFT_THRESHOLD:
            gaze_status = "LOOKING_RIGHT"
        else:
            gaze_status = "CENTER"
        

        # ── HEAD POSE ──────────────────────────────────────────────
        yaw, pitch, roll = estimate_head_pose(lm, w, h)

        if abs(roll) < ROLL_FLIP_LIMIT:
            yaw_buffer.append(yaw)
            pitch_buffer.append(pitch)

        smooth_yaw   = sum(yaw_buffer)   / len(yaw_buffer)   if yaw_buffer   else 0.0
        smooth_pitch = sum(pitch_buffer) / len(pitch_buffer) if pitch_buffer else 0.0
        corrected_yaw = smooth_yaw - YAW_OFFSET

        if corrected_yaw > YAW_THRESHOLD:
            head_status = "HEAD_TURN_LEFT"
        elif corrected_yaw < -YAW_THRESHOLD:
            head_status = "HEAD_TURN_RIGHT"
        elif smooth_pitch > PITCH_THRESHOLD:
            head_status = "HEAD_DOWN"
        else:
            head_status = "CENTER"

        # ── Draw iris dots ─────────────────────────────────────────
        cv2.circle(frame, li, 4, (0, 255, 255), -1)
        cv2.circle(frame, ri, 4, (0, 255, 255), -1)

    # ═══════════════════════════════════════════════════════════════
    #  VIOLATION CHECKS
    # ═══════════════════════════════════════════════════════════════
    check_and_log("NO_FACE",
        face_count == 0, now,
        "face_count=0")

    check_and_log("MULTIPLE_FACES",
        face_count >= 2, now,
        f"face_count={face_count}")

    check_and_log("GAZE_LEFT",
        gaze_status == "LOOKING_LEFT", now,
        f"ratio={avg_ratio:.2f}")

    check_and_log("GAZE_RIGHT",
        gaze_status == "LOOKING_RIGHT", now,
        f"ratio={avg_ratio:.2f}")

    check_and_log("HEAD_TURN",
        "HEAD_TURN" in head_status, now,
        f"yaw={smooth_yaw:.1f}")

    check_and_log("HEAD_DOWN",
        head_status == "HEAD_DOWN", now,
        f"pitch={smooth_pitch:.1f}")

    # ═══════════════════════════════════════════════════════════════
    #  STATUS DISPLAY
    # ═══════════════════════════════════════════════════════════════
    # Black header bar
    cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)

    # Face count
    face_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
    cv2.putText(frame, f"Faces: {face_count}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

    # Gaze status
    gaze_color = (0, 255, 0) if gaze_status == "CENTER" else (0, 0, 255)
    cv2.putText(frame, f"Gaze: {gaze_status}",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)

    # Head status
    head_color = (0, 255, 0) if head_status == "CENTER" else (0, 165, 255)
    cv2.putText(frame, f"Head: {head_status}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, head_color, 2)

    # Angle readouts bottom left
    cv2.putText(frame, f"yaw={smooth_yaw:+.0f} pitch={smooth_pitch:+.0f}",
                (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("AI Proctor v2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from violation_logger import end_session
end_session()
