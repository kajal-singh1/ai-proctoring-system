import cv2

# ── Load the pretrained DNN face detector ──────────────────────────
MODEL_FILE  = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"

net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
print("✅ Face detection model loaded.")

# ── Open webcam ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Running face detector... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]   # frame height and width in pixels

    # ── Prepare the frame for the neural network ───────────────────
    # The model expects a 300x300 blob with specific mean subtraction
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)  # mean BGR values to normalize
    )

    # ── Run forward pass (this is the actual inference) ───────────
    net.setInput(blob)
    detections = net.forward()

    face_count = 0

    # ── Loop through all detections ────────────────────────────────
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]   # how sure the model is

        if confidence > 0.5:   # only count if >50% confident
            face_count += 1

            # Bounding box coordinates are given as fractions (0–1)
            # Multiply by frame size to get pixel coordinates
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show confidence score above the box
            label = f"Face {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── Show face count on screen ──────────────────────────────────
    status_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
    status_text  = f"Faces detected: {face_count}"
    cv2.putText(frame, status_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")