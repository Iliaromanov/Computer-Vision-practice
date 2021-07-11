import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence

        self.mp_face_detect = mp.solutions.face_detection
        self.face_detection = self.mp_face_detect.FaceDetection(min_detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_face(self, img, draw=True):
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img)

        positions = []
        detections = results.detections

        if detections:
            for id, detection in enumerate(detections):
                bounding_box = detection.location_data.relative_bounding_box
                x, y = int(bounding_box.xmin * w), int(bounding_box.ymin * h)
                positions.append((id, x, y))

                if draw:
                    score = detection.score[0]
                    cv2.putText(img, str(int(float(score) * 100)) + '%', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.mp_draw.draw_detection(img, detection)

        return img, positions


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, positions = detector.find_face(img)
        print(positions)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()