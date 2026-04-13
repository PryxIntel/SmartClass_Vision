import cv2
import time
import shutil
from pathlib import Path
from src.database import add_student, delete_student, student_exists, get_student_info
from src.detector import FaceDetector
from utils.config import KNOWN_FACES_DIR, CAMERA_ID


def search_student_record(roll_number):
    """Searches the DB and returns the record and image count for the GUI."""
    record = get_student_info(roll_number)
    if record:
        student_dir = Path(KNOWN_FACES_DIR) / roll_number
        image_count = len(list(student_dir.glob("*.jpg"))) if student_dir.exists() else 0
        return True, record, image_count
    return False, None, 0


def delete_existing_student(roll_number):
    """Deletes a student and returns status to the GUI."""
    if not student_exists(roll_number):
        return False, f"Roll Number {roll_number} not found in database."

    delete_student(roll_number)
    student_dir = Path(KNOWN_FACES_DIR) / roll_number
    if student_dir.exists():
        shutil.rmtree(student_dir)
    return True, f"All records and images for {roll_number} deleted successfully."


def register_student(roll_number, name, degree, branch, section):
    """Handles the camera capture after receiving verified data from the GUI."""

    # 1. Double check database
    if student_exists(roll_number):
        return False, f"Roll Number {roll_number} is already registered!"

    if not add_student(roll_number, name, degree, branch, section):
        return False, "Database insertion failed."

    # 2. Setup folders and camera
    student_dir = Path(KNOWN_FACES_DIR) / roll_number
    student_dir.mkdir(parents=True, exist_ok=True)

    instructions = [
        "Look STRAIGHT at the camera",
        "Turn your head slightly LEFT",
        "Turn your head slightly RIGHT",
        "Tilt your head slightly UP",
        "Tilt your head slightly DOWN"
    ]

    cap = cv2.VideoCapture(CAMERA_ID)
    detector = FaceDetector()
    count = 0
    aborted = False
    BLUR_THRESHOLD = 80

    while count < 5:
        ret, frame = cap.read()
        if not ret: break

        clean_frame = frame.copy()
        h, w = frame.shape[:2]
        guide_w, guide_h = 300, 350
        guide_x1 = (w - guide_w) // 2
        guide_y1 = (h - guide_h) // 2
        guide_x2 = guide_x1 + guide_w
        guide_y2 = guide_y1 + guide_h

        processed_frame, cropped_faces = detector.detect_faces(frame)
        is_aligned = False
        msg = "Align your face inside the box"
        color = (0, 0, 255)

        if len(cropped_faces) == 1:
            fx1, fy1, fx2, fy2 = cropped_faces[0]["coords"]
            if fx1 > guide_x1 and fy1 > guide_y1 and fx2 < guide_x2 and fy2 < guide_y2:
                if (fx2 - fx1) > 80:
                    portrait_crop = clean_frame[guide_y1:guide_y2, guide_x1:guide_x2]
                    gray_crop = cv2.cvtColor(portrait_crop, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

                    if sharpness > BLUR_THRESHOLD:
                        is_aligned = True
                        msg = f"Perfect! (Sharpness: {int(sharpness)}) Press 'c'"
                        color = (0, 255, 0)
                    else:
                        msg = f"Hold still! Image blurry ({int(sharpness)}/{BLUR_THRESHOLD})"
                        color = (0, 165, 255)
                else:
                    msg = "Move closer to the camera"
            else:
                msg = "Center your face in the box"
        elif len(cropped_faces) > 1:
            msg = "Multiple faces detected! Only you should be in frame."

        cv2.rectangle(processed_frame, (guide_x1, guide_y1), (guide_x2, guide_y2), color, 3)
        cv2.putText(processed_frame, f"Pose {count + 1}/5: {instructions[count]}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, msg, (guide_x1, guide_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(processed_frame, "Press 'c' to Capture | Press 'q' to Abort", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Registration - Camera Capture", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            aborted = True
            break
        elif key == ord('c') and is_aligned:
            portrait_crop = clean_frame[guide_y1:guide_y2, guide_x1:guide_x2]
            img_path = student_dir / f"{roll_number}_{count}.jpg"
            cv2.imwrite(str(img_path), portrait_crop)
            count += 1
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    if aborted:
        delete_student(roll_number)
        if student_dir.exists(): shutil.rmtree(student_dir)
        return False, "Registration aborted by user. Data cleaned up."

    return True, f"Registration complete for {name}!"