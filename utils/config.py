import os
from pathlib import Path

# --- Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
KNOWN_FACES_DIR = BASE_DIR / "data" / "known_faces"
UNKNOWN_FACES_DIR = BASE_DIR / "data" / "unknown_faces"
ATTENDANCE_DIR = BASE_DIR / "data" / "attendance_records"

# Model Directories
YOLO_WEIGHTS = BASE_DIR / "models" / "yolo_weights" / "yolov8n-face.pt"
DEEPFACE_DIR = BASE_DIR / "models"

# --- Camera & YOLO Settings ---
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.60

# --- Business Logic Constraints ---
ATTENDANCE_WINDOW_MINUTES = 15