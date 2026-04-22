import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Objects to detect
objects = [ "mouse","dog", "cat", "cow", "horse", "sheep"]

# Start webcam
cap = cv2.VideoCapture(0)

# Check camera
if not cap.isOpened():
    print("Camera not working ❌")
    exit()

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            confidence = float(box.conf[0])  # 🔥 confidence

            # Filter objects + accuracy
            if label in objects and confidence > 0.5:
                print(f"Detected: {label} ({confidence:.2f})")

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Label + confidence text
                text = f"{label} {confidence:.2f}"

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # Put text
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show output
    cv2.imshow("Object Detection with Accuracy", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release
cap.release()
cv2.destroyAllWindows()