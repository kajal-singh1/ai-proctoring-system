# 🎓 AI Proctoring System

A real-time AI-powered exam proctoring system built with Python, OpenCV, 
and MediaPipe. Detects suspicious behavior during online exams using 
computer vision and deep learning — no GPU required.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 👤 Face Detection | Alerts when no face or multiple faces detected |
| 👁️ Gaze Tracking | Detects eye deviation using iris position ratio |
| 🔄 Head Pose | Detects head turns and looking down via solvePnP |
| 📋 Violation Logger | Timestamped CSV log with session IDs |
| 📊 Live Dashboard | Real-time Streamlit dashboard with charts |

---

## 🧠 Tech Stack

- **OpenCV** — webcam capture, DNN face detection, frame processing
- **MediaPipe** — 468-point facial landmark detection, iris tracking
- **solvePnP** — 3D head pose estimation using Euler angles
- **Streamlit** — live dashboard with auto-refresh
- **Plotly** — interactive violation timeline and breakdown charts
- **Python 3.10** — CPU only, runs on standard laptop hardware

---

## 📁 Project Structure
```
ai_proctor/
│
├── proctor_v2.py          # Main proctoring engine (all detectors)
├── violation_logger.py    # Session management + CSV logging
├── dashboard.py           # Live Streamlit dashboard
├── landmark_detector.py   # MediaPipe landmark visualization
├── gaze_tracker.py        # Standalone gaze tracker
├── head_pose.py           # Standalone head pose detector
├── face_detector.py       # Standalone DNN face detector
├── requirements.txt       # Python dependencies
├── run_proctor.bat        # One-click launcher (Windows)
│
└── models/
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

---

## ⚙️ Setup
```bash
# 1. Clone the repository
git clone https://github.com/kajal-singh1/ai-proctoring-system.git
cd ai-proctoring-system

# 2. Create virtual environment
python -m venv proctor_env
proctor_env\Scripts\activate       # Windows
source proctor_env/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download face detection model
python download_model.py
# Note: MediaPipe models download automatically on first run

# 5. Run the system
run_proctor.bat                    # Windows one-click
# OR manually:
streamlit run dashboard.py         # Terminal 1
python proctor_v2.py               # Terminal 2
```

---

## 🎯 Violations Detected

| Violation | Trigger | Threshold |
|-----------|---------|-----------|
| `NO_FACE` | Face absent from frame | > 1.5 seconds |
| `MULTIPLE_FACES` | 2+ faces detected | > 1.5 seconds |
| `GAZE_LEFT` | Eyes deviate left | Iris ratio < 0.45 |
| `GAZE_RIGHT` | Eyes deviate right | Iris ratio > 0.52 |
| `HEAD_TURN` | Head turned sideways | Yaw > 25 degrees |
| `HEAD_DOWN` | Head looking down | Pitch > 13 degrees |

---

## 📊 Dashboard

- **Live KPI cards** — total violations by type
- **Timeline chart** — violations per minute
- **Pie chart** — violation type breakdown  
- **Session selector** — filter by individual exam session
- **Risk level** — automatic Clean / Low / Medium / High rating

---

## 🧠 Deep Learning Concepts Used

- CNN inference with pretrained SSD face detector
- MediaPipe 468-point facial mesh with iris landmarks
- Iris Position Ratio for gaze direction estimation
- solvePnP + Rodrigues transform for 3D head pose
- Rolling average smoothing for noise reduction
- Per-user calibration for threshold tuning

---

## 👤 Author

Built by Kajal Singh as a portfolio project demonstrating
real-time computer vision and deep learning on CPU hardware.