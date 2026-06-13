import sqlite3
from pathlib import Path
from datetime import datetime
from utils.config import BASE_DIR

DB_PATH = BASE_DIR / "data" / "smartclass.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            roll_number TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            degree TEXT NOT NULL,
            year TEXT NOT NULL,
            branch TEXT NOT NULL,
            section TEXT NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def student_exists(roll_number):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM students WHERE roll_number = ?', (roll_number,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def get_student_info(roll_number):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT roll_number, name, gender, degree, year, branch, section, registered_at FROM students WHERE roll_number = ?', (roll_number,))
    result = cursor.fetchone()
    conn.close()
    return result

def add_student(roll_number, name, gender, degree, year, branch, section):
    init_db()
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO students (roll_number, name, gender, degree, year, branch, section, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (roll_number, name, gender, degree, year, branch, section, local_time))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        print(f"\n[ERROR] Roll Number {roll_number} is already registered!")
        return False

def delete_student(roll_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM students WHERE roll_number = ?', (roll_number,))
    conn.commit()
    conn.close()