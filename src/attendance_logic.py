import cv2
import time
import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from utils.config import ATTENDANCE_DIR, CLASS_PHOTOS_DIR, CAMERA_ID, BASE_DIR

DB_PATH = BASE_DIR / "data" / "smartclass.db"


def get_student_details(roll_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, branch, section FROM students WHERE roll_number = ?', (roll_number,))
    result = cursor.fetchone()
    conn.close()
    return result if result else ("Unknown", "Unknown", "Unknown")


def start_attendance():
    print("\n" + "=" * 40)
    print("   HIGH-SECURITY BATCH ATTENDANCE")
    print("=" * 40)

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if not recognizer.known_embeddings:
        print("[ERROR] AI Brain empty. Train model first!")
        return

    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_ID)

    # Force High Definition
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(2)

    NUM_SHOTS = 5
    REQUIRED_MATCHES = 2
    captured_frames = []

    print("[INFO] Initiating Multi-Shot Capture Sequence...")

    # PHASE 1: BATCH CAPTURE
    for i in range(NUM_SHOTS):
        start_wait = time.time()
        while time.time() - start_wait < 2.0:
            ret, frame = cap.read()
            if not ret: break

            display_frame = frame.copy()
            time_left = 2.0 - (time.time() - start_wait)

            cv2.putText(display_frame, f"High-Res Scan {i + 1}/{NUM_SHOTS}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 3)
            cv2.putText(display_frame, f"Hold Still: {int(time_left) + 1}...", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255), 3)

            cv2.imshow("SmartClass Vision - Security Scan", display_frame)
            cv2.waitKey(1)

        ret, frame = cap.read()
        if ret:
            captured_frames.append(frame.copy())
            photo_name = f"ClassAudit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_Shot{i + 1}.jpg"
            cv2.imwrite(str(CLASS_PHOTOS_DIR / photo_name), frame)
            print(f"[CAPTURE] Secured High-Res Image: {photo_name}")

            flash = np.ones(frame.shape, dtype="uint8") * 255
            cv2.imshow("SmartClass Vision - Security Scan", flash)
            cv2.waitKey(150)

    cap.release()

    # PHASE 2: AI PROCESSING
    processing_screen = np.zeros((400, 800, 3), dtype="uint8")
    cv2.putText(processing_screen, "Captures Complete.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(processing_screen, "AI is strictly verifying identities...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    cv2.imshow("SmartClass Vision - Security Scan", processing_screen)
    cv2.waitKey(100)

    print("\n[INFO] Cross-referencing captures with Database...")
    student_detections = {}

    for frame in captured_frames:
        processed_frame, cropped_faces = detector.detect_faces(frame)
        for face_data in cropped_faces:
            face_crop = face_data["image"]
            roll_number, confidence = recognizer.recognize(face_crop)

            if roll_number != "Unknown":
                student_detections[roll_number] = student_detections.get(roll_number, 0) + 1

    cv2.destroyAllWindows()

    # PHASE 3: CONSENSUS & EXPORT
    attendance_list = []
    timestamp = datetime.now().strftime("%H:%M:%S")

    print("\n" + "=" * 40)
    print("   STRICT VERIFICATION RESULTS")
    print("=" * 40)

    for roll, count in student_detections.items():
        name, branch, section = get_student_details(roll)

        if count >= REQUIRED_MATCHES:
            print(f"[VERIFIED] {name} ({roll}) matched in {count}/{NUM_SHOTS} shots -> PRESENT")
            attendance_list.append({
                "Roll Number": roll, "Name": name,
                "Branch": branch, "Section": section,
                "Time Marked": timestamp, "Status": "Present"
            })
        else:
            print(f"[REJECTED] {name} ({roll}) only seen in {count}/{NUM_SHOTS} shots -> FAKE/GLITCH DISCARDED")

    if attendance_list:
        filename = f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = ATTENDANCE_DIR / filename
        df = pd.DataFrame(attendance_list)
        df.to_csv(filepath, index=False)
        print(f"\n[SUCCESS] Final secure attendance saved to {filename}")
    else:
        print("\n[INFO] Zero students passed the strict security threshold.")