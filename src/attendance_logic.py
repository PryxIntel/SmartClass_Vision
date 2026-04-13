import cv2
import time
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from utils.config import ATTENDANCE_DIR, CAMERA_ID, ATTENDANCE_WINDOW_MINUTES, BASE_DIR

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
    print("   LIVE ATTENDANCE SYSTEM")
    print("=" * 40)

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    if not recognizer.known_embeddings:
        print("[ERROR] The Recognizer Brain is empty. Please register students and Train the model first!")
        return

    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    marked_students = set()
    attendance_list = []

    verification_counters = {}
    REQUIRED_FRAMES = 5

    # NEW: Create the CSV file immediately so the GUI can read it live
    filename = f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = ATTENDANCE_DIR / filename

    start_time = time.time()
    window_seconds = ATTENDANCE_WINDOW_MINUTES * 60

    print(f"[INFO] Camera booting up. Attendance window is open for {ATTENDANCE_WINDOW_MINUTES} minutes.")
    print(f"[INFO] Verification active: Requires {REQUIRED_FRAMES} matching frames.")
    print("[INFO] Press 'q' to end the class.")
    time.sleep(2)

    cap = cv2.VideoCapture(CAMERA_ID)

    PROCESS_EVERY_N_FRAMES = 4
    frame_counter = 0
    cached_drawing_data = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        elapsed_time = time.time() - start_time
        time_left = max(0, window_seconds - elapsed_time)
        minutes, seconds = divmod(int(time_left), 60)
        timer_text = f"Time Left: {minutes:02d}:{seconds:02d}"

        if time_left > 0:
            cv2.putText(frame, timer_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            attendance_active = True
        else:
            cv2.putText(frame, "ATTENDANCE CLOSED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            attendance_active = False

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            processed_frame, cropped_faces = detector.detect_faces(frame.copy())
            cached_drawing_data = []

            for face_data in cropped_faces:
                x1, y1, x2, y2 = face_data["coords"]
                face_crop = face_data["image"]

                roll_number, confidence = recognizer.recognize(face_crop)

                if roll_number != "Unknown":
                    name, branch, section = get_student_details(roll_number)
                    verification_counters[roll_number] = verification_counters.get(roll_number, 0) + 1

                    if verification_counters[roll_number] >= REQUIRED_FRAMES:
                        display_text = f"{name} (MARKED)"
                        color = (0, 255, 0)

                        if attendance_active and roll_number not in marked_students:
                            marked_students.add(roll_number)
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            attendance_list.append({
                                "Roll Number": roll_number, "Name": name,
                                "Branch": branch, "Section": section,
                                "Time Marked": timestamp, "Status": "Present"
                            })
                            print(f"[MARKED] Verified {name} ({roll_number}) at {timestamp}")

                            # NEW: Save to CSV instantly so the GUI Dashboard updates live
                            df = pd.DataFrame(attendance_list)
                            df.to_csv(filepath, index=False)
                    else:
                        current_count = verification_counters[roll_number]
                        display_text = f"Verifying {name}... ({current_count}/{REQUIRED_FRAMES})"
                        color = (0, 255, 255)
                else:
                    display_text = "Unknown"
                    color = (0, 0, 255)

                cached_drawing_data.append((x1, y1, x2, y2, color, display_text))

        for x1, y1, x2, y2, color, display_text in cached_drawing_data:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "Press 'q' to End Class", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.imshow("SmartClass Vision - Live Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Class ended. Camera closed.")