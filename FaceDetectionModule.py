import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, detectionCon=0.5, modelSelect=0):
        self.detectionCon = detectionCon
        self.modelSelect = modelSelect

        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.mpDraw = mp.solutions.drawing_utils

    def rescaleFrame(self, frame, scale=0.5):
        # Works for Images, Videos and Live Videos
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    def detectFace(self, frame):
        results = self.face.process(frame)
        if results.detections:
            for id, detection in enumerate(results.detections):
                h, w, c = frame.shape
                bBoxClass = detection.location_data.relative_bounding_box
                boundingBox = int(bBoxClass.xmin * w), int(bBoxClass.ymin * h), \
                    int(bBoxClass.width * w), int(bBoxClass.height * h)
                cv2.rectangle(frame, boundingBox, (255, 0, 255), 2)
                cv2.putText(frame, f"{int(detection.score[0] * 100)}%", (boundingBox[0], boundingBox[1] - 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255), 1)
        return frame

def main():
    ctime = 0
    ptime = 0

    cap = cv2.VideoCapture(0)

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        frame = detector.rescaleFrame(img, scale=1)
        frame = detector.detectFace(frame)

        ctime = time.time()
        fps = int(1 / (ctime - ptime))
        ptime = ctime

        cv2.putText(frame, str(fps), (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), thickness=1)
        cv2.imshow("Video", frame)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



