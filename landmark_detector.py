import cv2
import mediapipe as mp
import numpy as np

# ── Initialize MediaPipe Face Mesh ─────────────────────────────────
mp_face_mesh  = mp.solutions.face_mesh
mp_drawing    = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,          # detect up to 2 faces
    refine_landmarks=True,    # gives 478 points including iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe Face Mesh loaded.")

# ── Key landmark indices in MediaPipe's 468-point model ───────────
# These are the points we'll use for gaze tracking tomorrow
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # left eye outline
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # right eye outline
LEFT_IRIS     = [468, 469, 470, 471, 472]        # left iris (needs refine=True)
RIGHT_IRIS    = [473, 474, 475, 476, 477]        # right iris

NOSE_TIP      = 1      # single point for nose tip
CHIN          = 152    # single point for chin
LEFT_TEMPLE   = 234    # single point for left temple
RIGHT_TEMPLE  = 454    # single point for right temple

def get_pixel_coords(landmark, w, h):
    """Convert normalized landmark (0-1) to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)

# ── Open webcam ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("🎥 Landmark detector running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # MediaPipe needs RGB input
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # minor performance optimization
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    face_count = 0

    if results.multi_face_landmarks:
        face_count = len(results.multi_face_landmarks)

        for face_landmarks in results.multi_face_landmarks:

            # ── Draw full face mesh (all 468 points) ──────────────
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_style
                    .get_default_face_mesh_tesselation_style()
            )

            # ── Draw eye contours in green ─────────────────────────
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2)
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2)
            )

            # ── Highlight iris centers ─────────────────────────────
            for idx in [468, 473]:   # left iris center, right iris center
                lm = face_landmarks.landmark[idx]
                cx, cy = get_pixel_coords(lm, w, h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # ── Mark key points we'll use for gaze ────────────────
            for idx, label, color in [
                (NOSE_TIP,     "NOSE",  (0, 255, 255)),
                (CHIN,         "CHIN",  (255, 255, 0)),
                (LEFT_TEMPLE,  "L",     (0, 255, 0)),
                (RIGHT_TEMPLE, "R",     (255, 0, 0)),
            ]:
                lm = face_landmarks.landmark[idx]
                cx, cy = get_pixel_coords(lm, w, h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── Status bar ─────────────────────────────────────────────────
    status_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Faces: {face_count} | MediaPipe 468-point mesh",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("Landmark Detector - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")