import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFaceDetect = mp.solutions.face_detection
face_detection = mpFaceDetect.FaceDetection()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    detections = results.detections

    if detections:
        for detection in detections:
            mpDraw.draw_detection(img, detection)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
