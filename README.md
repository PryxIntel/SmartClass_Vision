\# 🎓 SmartClass Vision



\### AI-Powered Face Recognition Attendance Management System



SmartClass Vision is an intelligent attendance management platform that automates student identification and attendance tracking using computer vision and facial recognition technologies. The system leverages YOLOv8-based face detection and DeepFace-powered facial recognition to eliminate manual attendance processes and improve accuracy in educational environments.



\---



\## 🚀 Features



\* Real-time face detection using YOLOv8

\* Automated attendance marking through facial recognition

\* Student registration and profile management

\* Attendance report generation

\* Multi-shot face verification for improved accuracy

\* SQLite database integration for secure record storage

\* User-friendly graphical interface built with CustomTkinter

\* Attendance history and monitoring dashboard



\---



\## 🛠️ Technology Stack



\### Programming Language



\* Python



\### Computer Vision \& AI



\* YOLOv8

\* OpenCV

\* DeepFace

\* FaceNet512



\### Database



\* SQLite



\### GUI Framework



\* CustomTkinter



\### Supporting Libraries



\* NumPy

\* Pandas

\* Pillow



\---



\## 🏗️ System Architecture



```text

Camera Feed

&#x20;    │

&#x20;    ▼

YOLOv8 Face Detection

&#x20;    │

&#x20;    ▼

Face Cropping \& Processing

&#x20;    │

&#x20;    ▼

DeepFace Recognition Engine

&#x20;    │

&#x20;    ▼

Student Verification

&#x20;    │

&#x20;    ▼

Attendance Database (SQLite)

&#x20;    │

&#x20;    ▼

Dashboard \& Reports

```



\---



\## 📋 Core Modules



\### Student Registration



Registers new students by capturing facial data and storing embeddings for future recognition.



\### Face Detection



Detects faces in real-time using YOLOv8 face detection models.



\### Face Recognition



Matches detected faces with registered student profiles using DeepFace and FaceNet512 embeddings.



\### Attendance Management



Automatically marks attendance and stores records in the SQLite database.



\### Reporting System



Generates attendance reports and maintains attendance history for monitoring purposes.



\---



\## 💡 Key Highlights



\* Eliminates manual attendance procedures

\* Reduces proxy attendance possibilities

\* Improves attendance tracking efficiency

\* Provides automated record management

\* Supports scalable deployment in educational institutions



\---



\## 📂 Project Structure



```text

SmartClass\_Vision/

│

├── src/

│   ├── attendance\_logic.py

│   ├── database.py

│   ├── detector.py

│   ├── recognizer.py

│   └── registration.py

│

├── utils/

│   └── config.py

│

├── gui.py

├── requirements.txt

├── README.md

└── .gitignore

```



\---



\## ⚙️ Installation



```bash

git clone https://github.com/PryxIntel/SmartClass\_Vision.git

cd SmartClass\_Vision

pip install -r requirements.txt

python gui.py

```



\---



\## 🎯 Future Enhancements



\* Cloud database integration

\* Web-based administration portal

\* Student analytics dashboard

\* Email and SMS notifications

\* Multi-camera classroom support



\---



\## 👨‍💻 Author



\*\*Priyanshu Chauhan\*\*



B.Tech, Computer Science \& Engineering

Madan Mohan Malaviya University of Technology (MMMUT), Gorakhpur



GitHub: https://github.com/PryxIntel



