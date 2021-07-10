import cv2
import mediapipe as mp
import time

# Create video object
capture = cv2.VideoCapture(0) # using webcam number 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()



while True:
    success, img = capture.read()  # Get img frame

    cv2.imshow("Image", img)  # Display frame
    cv2.waitKey(1)  # Display each frame for 1ms
