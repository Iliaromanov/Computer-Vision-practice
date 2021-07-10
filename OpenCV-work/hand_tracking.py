import cv2
import mediapipe as mp
import time

# Create video object
capture = cv2.VideoCapture(0) # using webcam number 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_tracking_confidence=0.5,
                      min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = capture.read()  # Get img frame
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert captured img to RGB
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Calculate frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)  # Display text on frame

    cv2.imshow("Image", img)  # Display frame
    cv2.waitKey(1)  # Display each frame for 1ms

















