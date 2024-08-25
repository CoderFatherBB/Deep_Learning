import cv2
from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Load the video capture (you can replace 0 with a video file path)
cap = cv2.VideoCapture(0)

Total_object_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)
    # print(results)

    # Initialize the count of detected objects
    object_count = 0

    # Loop over the detections
    for result in results:
        # Extract the bounding box coordinates and label
        boxes = result.boxes  # Boxes object for bbox outputs
        # print(boxes)
        for box in boxes:
            x, y, w, h = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            label = 'Bottle_Cap'
            confidence = box.conf.item()

            # x = x - (w / 2)
            # y = y - (h / 2)

            # Draw a bounding box and label on the frame
            # cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Increment the count of detected objects
            object_count += 1
            Total_object_count += 1

    # Display the resulting frame with the object count
    cv2.putText(frame, f"Bottle caps detected: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Object Count: {Total_object_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
