import cv2
import time
from ultralytics import YOLO

# ==============================
# Phone camera stream settings
# ==============================
STREAM_URL = "http://192.168.1.2:4747/video"

# Example:
# STREAM_URL = "http://192.168.1.2:4747/video"

# ==============================
# Load YOLOv8 model
# ==============================
model = YOLO("yolov8n.pt")

# Open phone camera stream
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Could not connect to phone camera.")
    print("Check:")
    print("1. Phone and laptop are on the same Wi-Fi")
    print("2. DroidCam app is open")
    print("3. IP and port are correct")
    exit()

prev_time = 0
screenshot_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from phone camera.")
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    results = model(frame, conf=0.5, verbose=False)

    # Draw YOLO boxes
    annotated_frame = results[0].plot()

    # Count detected objects
    object_count = len(results[0].boxes)

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Add text overlay
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Objects: {object_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        "Press Q to quit | S to screenshot",
        (20, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Show result
    cv2.imshow("YOLOv8 Phone Camera Object Detector", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        screenshot_count += 1
        filename = f"screenshot_{screenshot_count}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Saved {filename}")

cap.release()
cv2.destroyAllWindows()