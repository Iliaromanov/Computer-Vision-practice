import cv2
import mediapipe as mp


class PoseDetector(mp.solutions.pose.Pose):
    def __init__(self, static_image_mode=False, upper_body_only=False, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode, upper_body_only, smooth_landmarks,
                                     min_detection_confidence, min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)

        lms = self.results.pose_landmarks

        if draw and lms:
            self.mpDraw.draw_landmarks(img, lms, self.mpPose.POSE_CONNECTIONS)

        return img

    def find_positions(self, img):
        lm_positions = []
        h, w, c = img.shape

        lms = self.results.pose_landmarks

        if lms:
            for id, lm in enumerate(lms.landmark):
                x_pos, y_pos = int(lm.x * w), int(lm.y * h)
                lm_positions.append((id, x_pos, y_pos))

        return lm_positions


def main():
    cap = cv2.VideoCapture(0)  # Capturing web cam 0 video
    # cap = cv2.VideoCapture('pathToVideo.mp4') # To run program on video file on disk
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.find_pose(img)

        cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Resized Window', 1500, 1000)

        cv2.imshow("Resized Window", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
