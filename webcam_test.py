import cv2

# Initialize webcam (0 = default/built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened successfully! Press Q to quit.")

while True:
    # Read one frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Write text onto the frame
    cv2.putText(
        frame,
        text="AI Proctor",
        org=(30, 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),   # Green in BGR
        thickness=2
    )

    # Display the frame in a window
    cv2.imshow("Webcam Feed", frame)

    # Wait 1ms for keypress — quit if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Always release resources when done
cap.release()
cv2.destroyAllWindows()
print("Webcam closed cleanly.")