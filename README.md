# Gesture-Controlled Face Replication

A computer vision project using MediaPipe and OpenCV that creates multiple blended copies of a detected face when a specific hand gesture is performed.

## How It Works

1. The webcam captures the live video.
2. MediaPipe detects the face and its landmarks.
3. MediaPipe also tracks both hands.
4. The midpoint between the index and middle fingers of each hand is calculated.
5. When these two midpoint positions come close together, the current face is captured.
6. Multiple resized and blended copies of the face are placed around the original face.
7. The result creates a real-time face replication effect.

## Instructions

1. Run the program.
2. Make sure your face and both hands are visible to the camera.
3. Bring the midpoint between the index and middle fingers of both hands together to trigger the effect.
4. The face replication effect will appear.
5. Press ESC to exit.
