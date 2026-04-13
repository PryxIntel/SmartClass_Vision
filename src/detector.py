import cv2
from ultralytics import YOLO
from utils.config import YOLO_WEIGHTS, CONFIDENCE_THRESHOLD


class FaceDetector:
    def __init__(self):
        """Initializes the YOLOv8 face detection model from the local file."""
        self.model = YOLO(YOLO_WEIGHTS)

    def detect_faces(self, frame):
        """Passes a video frame to YOLO to find faces."""
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        cropped_faces = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size != 0:
                    cropped_faces.append({
                        "coords": (x1, y1, x2, y2),
                        "image": face_crop
                    })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, cropped_faces