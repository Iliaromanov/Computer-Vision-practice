import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

"""
  ~~The 25 (upper-body) pose landmarks.~~
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32
"""

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   upper_body_only=False,
                   smooth_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Capturing web cam 0 video
# cap = cv2.VideoCapture('pathToVideo.mp4') # To run program on video file on disk

while True:
    success, img = cap.read()
    height, width, channels = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    lms = results.pose_landmarks

    if lms:
        mpDraw.draw_landmarks(img, lms, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(lms.landmark):
            x_pos, y_pos = int(lm.x * width), int(lm.y * height)

            if id == 0:
                cv2.circle(img, (x_pos, y_pos), 12, (0, 0, 255), cv2.FILLED) # Red circle on nose

            if id == 19:  # Left hand edge
                cv2.circle(img, (x_pos, y_pos), 25, (255, 255, 0), cv2.FILLED)


    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', 1500, 1000)

    cv2.imshow("Resized Window", img)
    cv2.waitKey(1)