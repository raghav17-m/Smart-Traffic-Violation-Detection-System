# 🚦 Smart Traffic Violation Detection System

An AI-powered traffic monitoring system that automatically detects, tracks, and records traffic violations in real time using computer vision and deep learning.

## 📌 Features

### 🚗 Vehicle Detection

* Real-time vehicle detection using **YOLOv8 / YOLOv11 (Ultralytics)**.
* Detects motorcycles, cars, buses, trucks, and other road vehicles.

### 🎯 Vehicle Tracking

* Multi-object tracking using:

  * **ByteTrack**
  * **BoT-SORT**
* Maintains unique IDs for each detected vehicle.

### 📊 Vehicle Counting

* Counts vehicles crossing predefined virtual lines.
* Supports entry/exit counting.

### 👥 Triple-Riding Detection

* Detects motorcycles carrying **more than two riders**.
* Flags triple-riding violations automatically.

### ⚡ Speed Tracking

* Estimates vehicle speed using pixel-based motion analysis.
* Configurable calibration line for real-world speed approximation.

### 🚨 Speed Violation Detection

* Detects vehicles exceeding a configurable speed limit.
* Generates violation events automatically.

### 📸 Violation Screenshot Capture

* Captures evidence images for:

  * Over-speeding
  * Triple-riding
  * Other configured violations

### 📝 CSV Violation Logging

* Automatically stores violation details:

  * Timestamp
  * Vehicle ID
  * Violation Type
  * Estimated Speed
  * Screenshot Path

### 🖥️ OpenCV GUI Dashboard

* Live monitoring dashboard displaying:

  * Video feed
  * Vehicle counts
  * Speed information
  * Violation alerts

### 📄 PDF Report Generator

* Generates a comprehensive PDF report at the end of each session.
* Includes:

  * Violation summaries
  * Statistics
  * Screenshots
  * Traffic analytics

---

## 🏗️ System Architecture

Input Video/Camera
↓
YOLO Detection
↓
Object Tracking
↓
Speed Estimation
↓
Violation Detection
↓
Evidence Capture & Logging
↓
Dashboard & PDF Report

---

## 🛠️ Tech Stack

| Component          | Technology           |
| ------------------ | -------------------- |
| Detection          | YOLOv8 / YOLOv11     |
| Tracking           | ByteTrack / BoT-SORT |
| Computer Vision    | OpenCV               |
| Deep Learning      | PyTorch              |
| Tracking Utilities | Supervision          |
| Data Logging       | CSV                  |
| Reporting          | PDF Generator        |
| Language           | Python               |

---

## 📂 Project Structure

```text
Smart-Traffic-Violation-Detection/
│
├── models/
│   └── yolov8.pt
│
├── screenshots/
│   └── violations/
│
├── reports/
│   └── session_reports/
│
├── logs/
│   └── violations.csv
│
├── config/
│   └── settings.yaml
│
├── main.py
├── detector.py
├── tracker.py
├── speed_estimator.py
├── violation_manager.py
├── report_generator.py
└── README.md
```

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Smart-Traffic-Violation-Detection.git

cd Smart-Traffic-Violation-Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / MacOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run with Video File

```bash
python main.py --source videos/traffic.mp4
```

### Run with Webcam

```bash
python main.py --source 0
```

### Configure Speed Limit

Edit configuration file:

```yaml
speed_limit: 60
```

---

## 📸 Sample Outputs

### Live Dashboard

* Vehicle Detection
* Vehicle Tracking IDs
* Speed Display
* Violation Alerts

### Violation Evidence

* Timestamped screenshots
* Vehicle information
* Speed information

### Generated Reports

* Session Summary
* Total Vehicles
* Total Violations
* Violation Categories
* Embedded Evidence Images

---

## 📈 Example Violation Log

| Timestamp | Vehicle ID | Violation     | Speed   |
| --------- | ---------- | ------------- | ------- |
| 10:12:03  | 15         | Over Speeding | 78 km/h |
| 10:15:21  | 22         | Triple Riding | N/A     |
| 10:18:40  | 31         | Over Speeding | 85 km/h |

---

## 🎯 Future Improvements

* Automatic Number Plate Recognition (ANPR)
* Helmet Detection
* Red-Light Violation Detection
* Wrong-Way Driving Detection
* Cloud Database Integration
* Real-Time Web Dashboard
* Email/SMS Alerts
* Multi-Camera Support

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed for intelligent traffic monitoring and road safety enforcement using Artificial Intelligence and Computer Vision.
