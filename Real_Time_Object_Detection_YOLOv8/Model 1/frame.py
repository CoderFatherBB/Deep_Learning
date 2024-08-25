import cv2
import os

# Ask for the output directory
output_dir = "Frames/"

# Check if the directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
video = cv2.VideoCapture('conveyer.mp4')

# Get the frames per second of the video
fps = video.get(cv2.CAP_PROP_FPS)

# Calculate the frame skip value to get 30 frames per second
frame_skip = int(fps / 30)

# Ensure frame_skip is at least 1
frame_skip = max(1, frame_skip)

# Initialize frame counter
frame_count = 0

while True:
    # Read the video frame by frame
    ret, frame = video.read()

    # If the frame was not retrieved, then we have reached the end of the video
    if not ret:
        break

    # Save each frame as an image in the specified directory
    if frame_count % frame_skip == 0:
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count}.jpg'), frame)

    # Increment the frame counter
    frame_count += 1

# Release the video file
video.release()

print(f'Video processed and frames saved as images in {output_dir}.')
