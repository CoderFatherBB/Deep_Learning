# import cv2
# from ultralytics import YOLO

# # Load your custom YOLOv8 model
# model = YOLO('runs/detect/train/weights/best.pt')

# # Load the video capture (you can replace 0 with a video file path)
# cap = cv2.VideoCapture('test2.mp4')

# # Total_object_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform object detection on the frame
#     results = model(frame, save=True)
#     # print(results)

#     # Initialize the count of detected objects
#     object_count = 0

#     # Loop over the detections
#     for result in results:
#         # Extract the bounding box coordinates and label
#         boxes = result.boxes  # Boxes object for bbox outputs
#         # print(boxes)
#         for box in boxes:
#             x, y, w, h = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
#             label = 'Bottle_Cap'
#             confidence = box.conf.item()

#             # x = x - (w / 2)
#             # y = y - (h / 2)

#             # Draw a bounding box and label on the frame
#             # cv2.rectangle(frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)
#             cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

#             text = f"{label}: {confidence:.2f}"
#             cv2.putText(frame, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Increment the count of detected objects
#             object_count += 1
#             # Total_object_count += 1

#     # Display the resulting frame with the object count
#     cv2.putText(frame, f"Bottle caps detected: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     # cv2.putText(frame, f"Total Object Count: {Total_object_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow("Frame", frame)

#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(30) & 0xFF == ord("q"):
#         break

# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
print(model.names)
cap = cv2.VideoCapture("test2.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
# region_points = [(500, 400), (2000, 404), (2000, 360), (500, 360)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=True)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()