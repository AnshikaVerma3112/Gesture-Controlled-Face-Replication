import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from mediapipe.tasks.python.vision import RunningMode
import numpy as np

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

face_model = "face_landmarker.task"

face_base = python.BaseOptions(
    model_asset_path=face_model
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)
clone_mode = False


last_point = None

frame_count = 0
points = []
frame_timestamp = 0
color = (0, 0, 255)      # red
while True:
    frame_timestamp += 33
    frame_count += 1
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    face_result = face_detector.detect(mp_image)
    result = detector.detect_for_video(mp_image,frame_timestamp)
    h, w, _ = frame.shape

    face_crop = None

    if face_result.face_landmarks:

        face = face_result.face_landmarks[0]

        xs = [lm.x for lm in face]
        ys = [lm.y for lm in face]

        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)

        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        pad_x = 120
        pad_y_top = 50
        pad_y_bottom = 300

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y_top)

        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y_bottom)

        face_crop = frame[y1:y2, x1:x2].copy()
        center_face = face_crop.copy()
        if face_crop.size == 0:
            continue
        mask = np.zeros(face_crop.shape[:2], dtype=np.uint8)

        cv2.ellipse(
            mask,
            (mask.shape[1]//2, mask.shape[0]//2),
            (
                int(mask.shape[1]*0.4),
                int(mask.shape[0]*0.45)
            ),

            0,0,360,
            255,
            -1
        )
        

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)


        if result.hand_landmarks and len(result.hand_landmarks) == 2:
        
            hand1 = result.hand_landmarks[0]
            hand2 = result.hand_landmarks[1]

            h, w, _ = frame.shape

# Hand 1
            index1 = hand1[7]      # index fingertip
            middle1 = hand1[11]    # middle fingertip

            mid1_x = ((index1.x + middle1.x) / 2) * w
            mid1_y = ((index1.y + middle1.y) / 2) * h

# Hand 2
            index2 = hand2[7]
            middle2 = hand2[11]

            mid2_x = ((index2.x + middle2.x) / 2) * w
            mid2_y = ((index2.y + middle2.y) / 2) * h

# Draw midpoint markers
            cv2.circle(frame, (int(mid1_x), int(mid1_y)), 8, (255,100,100), -1)
            cv2.circle(frame, (int(mid2_x), int(mid2_y)), 8, (100,100,255), -1)

# Connect them
            cv2.line(frame,(int(mid1_x), int(mid1_y)),(int(mid2_x), int(mid2_y)),(0,0,0),2)

            distance = np.sqrt((mid1_x - mid2_x)**2 +(mid1_y - mid2_y)**2)

            cv2.putText(frame,f"{int(distance)}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

            if distance < 30 and not clone_mode:
                clone_mode = True
                saved_crop = face_crop.copy()

    if clone_mode and face_crop is not None:

        face_cx = (x1 + x2) // 2
        face_cy = (y1 + y2) // 2

        offsets = [
    (-300,-180),
    (-180,-150),
    (0,-170),
    (180,-150),
    (300,-180),

    (-250,-20),
    (-100,-40),
    (100,-40),
    (250,-20),

    (-180,120),
    (0,150),
    (180,120)
]

        scales = [
            0.35,
            0.4,
            0.45,
            0.4,
            0.35,

            0.55,
            0.65,
            0.65,
            0.55,

            0.8,
            0.9,
            0.8
            ]


        for (dx, dy), scale in zip(offsets, scales):

            clone = cv2.resize(face_crop,None,fx=scale,fy=scale)

            ch, cw = clone.shape[:2]

            px = int(face_cx + dx - cw/2)
            py = int(face_cy + dy - ch/2)

            if (
                px >= 0 and
                py >= 0 and
                px + cw < frame.shape[1] and
                py + ch < frame.shape[0]
            ):
                h_clone, w_clone = clone.shape[:2]

                if (
                    px >= 0 and
                    py >= 0 and
                    px + w_clone <= frame.shape[1] and
                    py + h_clone <= frame.shape[0]
                ):
                    roi = frame[py:py+h_clone, px:px+w_clone]

                    alpha = 0.65

                    blended = cv2.addWeighted(
                        clone,
                        alpha,
                        roi,
                        1 - alpha,
                        0
                    )

                    mask_resized = cv2.resize(mask, (w_clone, h_clone))
                    mask_blur = cv2.GaussianBlur(mask_resized, (31,31), 0)

                    alpha = mask_blur.astype(float)/255.0
                    alpha = cv2.merge([alpha, alpha, alpha])

                    roi = frame[py:py+h_clone, px:px+w_clone]

                    blended = (
                        clone.astype(float) * alpha +
                        roi.astype(float) * (1-alpha)
                    ).astype(np.uint8)

                    frame[py:py+h_clone, px:px+w_clone] = blended
        roi = frame[y1:y2, x1:x2]

        mask_blur = cv2.GaussianBlur(mask, (41,41), 0)

        alpha = mask_blur.astype(float) / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])

        blended = (
            center_face.astype(float) * alpha +
            roi.astype(float) * (1 - alpha)
        ).astype(np.uint8)

        frame[y1:y2, x1:x2] = blended

        
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    if key == 27:
        break
    cv2.imshow("Hand Tracking", frame)
    


cap.release()
cv2.destroyAllWindows() 

            



