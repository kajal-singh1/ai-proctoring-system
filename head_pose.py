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
print("Head pose detector loaded.")

# ── 6 landmark indices for pose estimation ─────────────────────────
# These specific points give the best 3D pose stability
POSE_LANDMARKS = {
    "nose_tip" : 1,
    "chin" : 152,
    "left_eye" : 33, 
    "right_eye" : 263,
    "left_mouth" : 61,
    "right_mouth" : 291
}

# ── 3D reference points (generic human face model in mm) ──────────
# These are standard facial geometry coordinates used in pose estimation
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # nose tip
    (0.0,   -63.6, -12.5),   # chin
    (-43.3,  32.7, -26.0),   # left eye corner
    (43.3,   32.7, -26.0),   # right eye corner
    (-28.9, -28.9, -24.1),   # left mouth
    (28.9,  -28.9, -24.1),   # right mouth
], dtype=np.float64)

# ── Head pose thresholds (degrees) ────────────────────────────────
YAW_THRESHOLD   = 25
PITCH_THRESHOLD = 13
YAW_OFFSET      = 5    # your natural straight yaw is ~+5
ROLL_FLIP_LIMIT = 90   # ignore frames where roll flips to ~180

# ── Smoothing buffers ──────────────────────────────────────────────
yaw_buffer   = deque(maxlen=10)
pitch_buffer = deque(maxlen=10)

# ── Violation cooldown ─────────────────────────────────────────────
ALERT_SECONDS = 1.5
COOLDOWN_SECONDS = 5
last_logged = {"HEAD_TURN": 0, "HEAD_DOWN": 0}
violation_start = {"HEAD_TURN": None, "HEAD_DOWN": None}

def get_2d_points(landmarks, h, w):
    """Extract 6 key landmark pixel coordinates"""
    pts = []
    for name, idx in POSE_LANDMARKS.items():
        lm = landmarks[idx]
        pts.append((int(lm.x * w), int(lm.y * h)))
    return np.array(pts, dtype=np.float64)

def estimate_head_pose(image_points, w, h):
    """
    Use solvedPnP to estimate head rotation angles.
    Returns (yaw, pitch, roll) in degrees.
    """

    # Camera internals — estimated from frame size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # no lens distortion

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0
    
    # Convert rotation vector → rotation matrix → Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Simpler Euler extraction
    sy = np.sqrt(rotation_mat[0,0]**2 + rotation_mat[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2( rotation_mat[2,1], rotation_mat[2,2]))
        yaw   = np.degrees(np.arctan2(-rotation_mat[2,0], sy))
        roll  = np.degrees(np.arctan2( rotation_mat[1,0], rotation_mat[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-rotation_mat[1,2], rotation_mat[1,1]))
        yaw   = np.degrees(np.arctan2(-rotation_mat[2,0], sy))
        roll  = 0.0

    # Normalize pitch: values near 180/-180 mean looking straight
    # Shift so straight ahead = 0
    if pitch > 90:
        pitch = pitch - 180
    elif pitch < -90:
        pitch = pitch + 180

    return yaw, pitch, roll

# ── Initialize logger ──────────────────────────────────────────────
init_logger()

# ── Open webcam ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("🎥 Head pose detector running. Press Q to quit.")
print("Try turning your head LEFT, RIGHT, and looking DOWN.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    now  = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    yaw   = 0.0
    pitch = 0.0
    roll  = 0.0
    head_status = "NO_FACE"

    if results.multi_face_landmarks:
        landmarks    = results.multi_face_landmarks[0].landmark
        image_points = get_2d_points(landmarks, h, w)

        yaw, pitch, roll = estimate_head_pose(image_points, w, h)

        # ── Smooth the angles ──────────────────────────────────────
       # Skip flipped frames — roll near 180 means bad detection
        if abs(roll) < ROLL_FLIP_LIMIT:
            yaw_buffer.append(yaw)
            pitch_buffer.append(pitch)

        smooth_yaw   = sum(yaw_buffer)   / len(yaw_buffer) if yaw_buffer else 0.0
        smooth_pitch = sum(pitch_buffer) / len(pitch_buffer) if pitch_buffer else 0.0

        # ── Classify head direction ────────────────────────────────
        corrected_yaw = smooth_yaw - YAW_OFFSET
        if corrected_yaw > YAW_THRESHOLD:
            head_status = "HEAD_TURN"
        elif corrected_yaw < -YAW_THRESHOLD:
            head_status = "HEAD_TURN"
        elif smooth_pitch > PITCH_THRESHOLD:
            head_status = "HEAD_DOWN"
        else:
            head_status = "CENTER"

        # ── Draw the 6 pose points ─────────────────────────────────
        for i, (x, y) in enumerate(image_points.astype(int)):
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

        # ── Show angle readouts ────────────────────────────────────
        cv2.putText(frame, f"Yaw:   {smooth_yaw:+.1f} deg",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Pitch: {smooth_pitch:+.1f} deg",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Roll:  {roll:+.1f} deg",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # ── Violation logic ────────────────────────────────────────────
    current_violation = head_status if head_status in ["HEAD_TURN","HEAD_DOWN"] else None

    for v_type in ["HEAD_TURN", "HEAD_DOWN"]:
        if current_violation == v_type:
            if violation_start[v_type] is None:
                violation_start[v_type] = now
            duration  = now - violation_start[v_type]
            time_since_last = now - last_logged[v_type]
            if duration >= ALERT_SECONDS and time_since_last >= COOLDOWN_SECONDS:
                log_violation(v_type, f"yaw={yaw:+.1f} pitch={pitch:+.1f}")
                last_logged[v_type] = now
        else:
            violation_start[v_type] = None

    # ── Status display ─────────────────────────────────────────────
    if head_status == "CENTER":
        status_text  = "✓ Head: CENTER"
        status_color = (0, 255, 0)
    elif head_status == "HEAD_TURN":
        direction = "LEFT" if (smooth_yaw - YAW_OFFSET) < 0 else "RIGHT"
        status_text  = f"⚠ HEAD TURN {direction}"
        status_color = (0, 0, 255)
    elif head_status == "HEAD_DOWN":
        status_text  = "⚠ HEAD DOWN"
        status_color = (0, 165, 255)
    else:
        status_text  = "NO FACE"
        status_color = (128, 128, 128)

    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Head Pose Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Head pose session ended.")