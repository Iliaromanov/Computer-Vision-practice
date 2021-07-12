import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
# Editing default drawing specs
draw_specs = mp_draw.DrawingSpec(circle_radius=1, thickness=1)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    landmarks = results.multi_face_landmarks
    if landmarks:
        for face_lms in landmarks:
            mp_draw.draw_landmarks(img, face_lms, mp_face_mesh.FACE_CONNECTIONS, draw_specs, draw_specs)

    cv2.imshow("Face Mesh", img)
    cv2.waitKey(1)
