import cv2
import mediapipe as mp
import time

"""
  ~~The 21 hand landmarks.~~
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20
"""

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
    height, width, channels = img.shape  # Get dimensions and channels of current frame image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert captured img to RGB
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                x_pos, y_pos = int(lm.x * width), int(lm.y * height)  # Get x and y coordinate of landmark
                if id == 8:
                    cv2.circle(img, (x_pos, y_pos), 20, (0, 255, 255), cv2.FILLED)

                print(f"id:{id} x:{x_pos} y:{y_pos}")

            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Calculate frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)  # Display text on frame

    cv2.imshow("Image", img)  # Display frame
    cv2.waitKey(1)  # Display each frame for 1ms

















