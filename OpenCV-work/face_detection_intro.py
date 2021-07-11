import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFaceDetect = mp.solutions.face_detection
face_detection = mpFaceDetect.FaceDetection()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    detections = results.detections

    if detections:
        for id, detection in enumerate(detections):
            score = detection.score[0]
            bounding_box = detection.location_data.relative_bounding_box
            x, y = int(bounding_box.xmin * w), int(bounding_box.ymin * h)

            cv2.putText(img, str(int(float(score)*100))+'%', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mpDraw.draw_detection(img, detection)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
