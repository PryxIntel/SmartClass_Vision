import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import customtkinter as ctk
from tkinter import messagebox
import sqlite3
from datetime import datetime
from pathlib import Path
import threading
import pandas as pd  # <-- Required for parsing the roster

from src.database import init_db
from src.registration import register_student, delete_existing_student, search_student_record
from src.recognizer import FaceRecognizer
from src.attendance_logic import start_attendance
from utils.config import BASE_DIR, ATTENDANCE_DIR

DB_PATH = BASE_DIR / "data" / "smartclass.db"
EMB_PATH = BASE_DIR / "data" / "embeddings.pkl"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SmartClassApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        init_db()

        self.title("SmartClass Vision - Command Center")
        self.geometry("1050x650")  # Made slightly larger to accommodate the table
        self.resizable(False, False)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT SIDEBAR
        # ==========================================
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SMARTCLASS", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))
        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="Vision System v2.0", font=ctk.CTkFont(size=12),
                                           text_color="gray")
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        ctk.CTkButton(self.sidebar_frame, text="Register Student", command=self.open_registration_window).grid(row=2,
                                                                                                               column=0,
                                                                                                               padx=20,
                                                                                                               pady=10,
                                                                                                               sticky="ew")
        ctk.CTkButton(self.sidebar_frame, text="Search Record", command=self.search_student, fg_color="#2b2b2b",
                      hover_color="#404040").grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkButton(self.sidebar_frame, text="Delete Student", command=self.delete_student, fg_color="#8b0000",
                      hover_color="#a52a2a").grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.btn_train = ctk.CTkButton(self.sidebar_frame, text="Train AI Brain", command=self.train_model,
                                       fg_color="#cc7000", hover_color="#e68a00")
        self.btn_train.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        self.btn_attendance = ctk.CTkButton(self.sidebar_frame, text="LIVE ATTENDANCE",
                                            command=self.start_live_attendance, height=50,
                                            font=ctk.CTkFont(weight="bold"), fg_color="#006400", hover_color="#008000")
        self.btn_attendance.grid(row=7, column=0, padx=20, pady=(10, 30), sticky="ew")

        # ==========================================
        # RIGHT MAIN AREA (Tabview)
        # ==========================================
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Header: Clock
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.time_label = ctk.CTkLabel(self.header_frame, text="--:--:--", font=ctk.CTkFont(size=24, weight="bold"),
                                       text_color="#00a8cc")
        self.time_label.grid(row=0, column=1, sticky="e")
        self.date_label = ctk.CTkLabel(self.header_frame, text="----", font=ctk.CTkFont(size=14), text_color="gray")
        self.date_label.grid(row=1, column=1, sticky="e", pady=(0, 10))

        # Tabview Setup
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, sticky="nsew")

        self.tab_overview = self.tabview.add("System Overview")
        self.tab_roster = self.tabview.add("Today's Roster")

        # --- TAB 1: SYSTEM OVERVIEW ---
        self.tab_overview.grid_columnconfigure((0, 1, 2), weight=1)

        self.card_students = ctk.CTkFrame(self.tab_overview, height=120, corner_radius=15)
        self.card_students.grid(row=0, column=0, padx=10, pady=20, sticky="nsew")
        ctk.CTkLabel(self.card_students, text="Total Students", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(20, 5))
        self.lbl_total_students = ctk.CTkLabel(self.card_students, text="0", font=ctk.CTkFont(size=36, weight="bold"),
                                               text_color="#00ffcc")
        self.lbl_total_students.pack()

        self.card_ai = ctk.CTkFrame(self.tab_overview, height=120, corner_radius=15)
        self.card_ai.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")
        ctk.CTkLabel(self.card_ai, text="AI Brain Status", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 5))
        self.lbl_ai_status = ctk.CTkLabel(self.card_ai, text="Checking...", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_ai_status.pack()

        self.card_attendance = ctk.CTkFrame(self.tab_overview, height=120, corner_radius=15)
        self.card_attendance.grid(row=0, column=2, padx=10, pady=20, sticky="nsew")
        ctk.CTkLabel(self.card_attendance, text="Attendance Sheets", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(20, 5))
        self.lbl_sheets = ctk.CTkLabel(self.card_attendance, text="0 Generated",
                                       font=ctk.CTkFont(size=24, weight="bold"), text_color="#ffcc00")
        self.lbl_sheets.pack()

        # Progress Bar Area
        self.progress_frame = ctk.CTkFrame(self.tab_overview, corner_radius=15)
        self.progress_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=20, sticky="nsew")
        ctk.CTkLabel(self.progress_frame, text="Today's Attendance Progress",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))
        self.lbl_progress_text = ctk.CTkLabel(self.progress_frame, text="0 / 0 Present", font=ctk.CTkFont(size=14),
                                              text_color="gray")
        self.lbl_progress_text.pack()
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, width=500, height=20, progress_color="#00ffcc")
        self.progress_bar.pack(pady=15)
        self.progress_bar.set(0)

        # --- TAB 2: LIVE ROSTER ---
        self.tab_roster.grid_rowconfigure(1, weight=1)
        self.tab_roster.grid_columnconfigure(0, weight=1)

        # Roster Controls
        self.roster_ctrl_frame = ctk.CTkFrame(self.tab_roster, fg_color="transparent")
        self.roster_ctrl_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(self.roster_ctrl_frame, text="Live Roster Comparison",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(self.roster_ctrl_frame, text="Refresh Data", command=self.populate_roster, width=100).pack(
            side="right")

        # Roster Table (Scrollable)
        self.table_frame = ctk.CTkScrollableFrame(self.tab_roster)
        self.table_frame.grid(row=1, column=0, sticky="nsew")
        self.table_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Start background processes
        self.update_clock()
        self.refresh_dashboard_metrics()

    # --- Dashboard & Roster Logic ---
    def update_clock(self):
        now = datetime.now()
        self.time_label.configure(text=now.strftime("%I:%M:%S %p"))
        self.date_label.configure(text=now.strftime("%A, %B %d, %Y"))
        self.after(1000, self.update_clock)

    def refresh_dashboard_metrics(self):
        # 1. Total Students
        total_students = 0
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students")
            total_students = cursor.fetchone()[0]
            conn.close()
            self.lbl_total_students.configure(text=str(total_students))
        except:
            self.lbl_total_students.configure(text="0")

        # 2. AI Brain
        if EMB_PATH.exists():
            self.lbl_ai_status.configure(text="Trained ✅", text_color="#00ffcc")
        else:
            self.lbl_ai_status.configure(text="Untrained ❌", text_color="#ff4444")

        # 3. Sheets & Progress
        today = datetime.now().strftime("%Y%m%d")
        present_count = 0

        if ATTENDANCE_DIR.exists():
            sheets = list(ATTENDANCE_DIR.glob(f"Attendance_{today}*.csv"))
            self.lbl_sheets.configure(text=f"{len(sheets)} Generated")

            # Read the latest sheet to update progress bar
            if sheets:
                latest_sheet = max(sheets, key=os.path.getctime)
                try:
                    df = pd.read_csv(latest_sheet)
                    present_count = len(df)
                except:
                    pass

        self.lbl_progress_text.configure(text=f"{present_count} / {total_students} Present")
        if total_students > 0:
            self.progress_bar.set(present_count / total_students)
        else:
            self.progress_bar.set(0)

        # Auto-update the roster tab
        self.populate_roster()

        # Keep updating metrics automatically every 5 seconds if camera is running
        if self.btn_attendance.cget("state") == "disabled":
            self.after(5000, self.refresh_dashboard_metrics)

    def populate_roster(self):
        """Builds the Present/Absent list dynamically."""
        # 1. Clear old table
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        # Headers
        headers = ["Roll Number", "Name", "Branch", "Status", "Time Marked"]
        for col, text in enumerate(headers):
            ctk.CTkLabel(self.table_frame, text=text, font=ctk.CTkFont(weight="bold")).grid(row=0, column=col, pady=5,
                                                                                            sticky="ew")

        # 2. Get All DB Students
        all_students = {}
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT roll_number, name, branch FROM students")
            for r, n, b in cursor.fetchall():
                all_students[str(r)] = {"name": n, "branch": b, "status": "ABSENT", "time": "--"}
            conn.close()
        except Exception as e:
            pass

        # 3. Overwrite with Today's Present Students
        today = datetime.now().strftime("%Y%m%d")
        if ATTENDANCE_DIR.exists():
            sheets = list(ATTENDANCE_DIR.glob(f"Attendance_{today}*.csv"))
            if sheets:
                latest_sheet = max(sheets, key=os.path.getctime)
                try:
                    df = pd.read_csv(latest_sheet)
                    for index, row in df.iterrows():
                        roll = str(row['Roll Number'])
                        if roll in all_students:
                            all_students[roll]["status"] = "PRESENT"
                            all_students[roll]["time"] = row['Time Marked']
                except:
                    pass

        # 4. Draw Rows
        row_idx = 1
        for roll, data in all_students.items():
            ctk.CTkLabel(self.table_frame, text=roll).grid(row=row_idx, column=0, pady=5)
            ctk.CTkLabel(self.table_frame, text=data["name"]).grid(row=row_idx, column=1, pady=5)
            ctk.CTkLabel(self.table_frame, text=data["branch"]).grid(row=row_idx, column=2, pady=5)

            # Color Status
            status_color = "#00ffcc" if data["status"] == "PRESENT" else "#ff4444"
            ctk.CTkLabel(self.table_frame, text=data["status"], text_color=status_color,
                         font=ctk.CTkFont(weight="bold")).grid(row=row_idx, column=3, pady=5)

            ctk.CTkLabel(self.table_frame, text=data["time"]).grid(row=row_idx, column=4, pady=5)
            row_idx += 1

    # --- Action Methods ---
    def open_registration_window(self):
        reg_window = ctk.CTkToplevel(self)
        reg_window.title("Register Student")
        reg_window.geometry("400x500")
        reg_window.attributes('-topmost', True)

        ctk.CTkLabel(reg_window, text="Student Details", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=15)
        roll_entry = ctk.CTkEntry(reg_window, placeholder_text="Roll Number (Numbers Only)", width=250)
        roll_entry.pack(pady=10)
        name_entry = ctk.CTkEntry(reg_window, placeholder_text="Full Name", width=250)
        name_entry.pack(pady=10)
        degree_var = ctk.StringVar(value="Select Degree")
        ctk.CTkOptionMenu(reg_window, variable=degree_var, values=["BTECH", "MTECH", "PHD", "BBA"], width=250).pack(
            pady=10)
        branch_var = ctk.StringVar(value="Select Branch")
        ctk.CTkOptionMenu(reg_window, variable=branch_var,
                          values=["CSE", "MECHANICAL", "CHEMICAL", "ELECTRICAL", "ECE", "ECE-IOT", "IT", "NONE"],
                          width=250).pack(pady=10)
        section_var = ctk.StringVar(value="Select Section")
        ctk.CTkOptionMenu(reg_window, variable=section_var, values=["A", "B", "C", "D", "NONE"], width=250).pack(
            pady=10)

        def submit_form():
            roll, name, degree = roll_entry.get().strip(), name_entry.get().strip(), degree_var.get()
            branch = branch_var.get() if branch_var.get() != "NONE" else ""
            section = section_var.get() if section_var.get() != "NONE" else ""
            if not roll.isdigit() or not name or degree == "Select Degree":
                messagebox.showerror("Validation Error", "Please fill out Roll Number, Name, and Degree.")
                return
            reg_window.withdraw()
            success, msg = register_student(roll, name, degree, branch, section)

            if success:
                messagebox.showinfo("Success", msg)
                self.refresh_dashboard_metrics()
                reg_window.destroy()
            else:
                messagebox.showerror("Error", msg)
                reg_window.deiconify()

        ctk.CTkButton(reg_window, text="Open Camera & Capture", command=submit_form, fg_color="#006400",
                      hover_color="#008000").pack(pady=20)

    def search_student(self):
        dialog = ctk.CTkInputDialog(text="Enter Roll Number to search:", title="Search Database")
        roll = dialog.get_input()
        if roll and roll.isdigit():
            found, record, img_count = search_student_record(roll)
            if found:
                info = (f"Roll Number: {record[0]}\nName: {record[1]}\nDegree: {record[2]}\n"
                        f"Branch: {record[3]}\nSection: {record[4]}\nRegistered: {record[5]}\nFace Images: {img_count}/5")
                messagebox.showinfo("Record Found", info)
            else:
                messagebox.showerror("Not Found", f"No record found for Roll Number {roll}")

    def delete_student(self):
        dialog = ctk.CTkInputDialog(text="Enter Roll Number to permanently DELETE:", title="Delete Record")
        roll = dialog.get_input()
        if roll and roll.isdigit():
            success, msg = delete_existing_student(roll)
            if success:
                messagebox.showinfo("Deleted", msg)
                self.refresh_dashboard_metrics()
            else:
                messagebox.showerror("Error", msg)

    def train_model(self):
        self.btn_train.configure(text="Training... Please Wait", state="disabled")
        self.update()

        recognizer = FaceRecognizer()
        recognizer.train_system()

        self.btn_train.configure(text="Train AI Brain", state="normal")
        self.refresh_dashboard_metrics()
        messagebox.showinfo("Success", "AI Recognizer Model trained successfully!")

    def start_live_attendance(self):
        self.btn_attendance.configure(text="CAMERA RUNNING...", state="disabled")
        self.update()
        threading.Thread(target=self._run_camera_thread, daemon=True).start()
        # Starts a loop to update the roster live while camera is open
        self.after(5000, self.refresh_dashboard_metrics)

    def _run_camera_thread(self):
        start_attendance()
        self.after(0, self._camera_finished)

    def _camera_finished(self):
        self.btn_attendance.configure(text="LIVE ATTENDANCE", state="normal")
        self.refresh_dashboard_metrics()
        messagebox.showinfo("Class Ended", "Attendance stopped and Roster updated!")


if __name__ == "__main__":
    app = SmartClassApp()
    app.mainloop()