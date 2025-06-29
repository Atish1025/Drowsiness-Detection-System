   ### 🚗 Driver Drowsiness Detection System

This project is a real-time Driver Drowsiness Detection System that uses **Computer Vision** to monitor eye movement and alert the driver if signs of drowsiness are detected. It leverages **Dlib**, **MediaPipe**, and **OpenCV** to analyze facial landmarks and eye aspect ratios (EAR), providing an audio warning when drowsiness is detected.

---

### 🔧 Features

* 👁️ Eye Aspect Ratio (EAR) based drowsiness detection using **Dlib**
* 🎯 Face Mesh-based approach using **MediaPipe** for improved accuracy
* 🔊 Real-time audio alerts using **Pygame**
* 📸 Camera selection and configuration (works with phone/laptop camera)
* 🧠 Works with pre-trained facial landmark models (`shape_predictor_68_face_landmarks.dat`)

---

### 📦 Tech Stack

* Python
* OpenCV
* Dlib / MediaPipe
* Pygame
* Imutils

---

### 🎯 How It Works

1. Captures live video stream from the camera.
2. Detects facial landmarks and extracts eye regions.
3. Calculates Eye Aspect Ratio (EAR) or eye radius.
4. If EAR or radius drops below a threshold for a number of frames → triggers a **drowsiness alert**.

---

### 📁 Files Overview

* `Drowsiness_Detection.py` – Dlib-based EAR model
* `Drowsiness_Detection_1.py` – MediaPipe face mesh model
* `Drowsiness_Detection_2.py` – Dlib model with camera selector

---

### 🛠️ Setup

```bash
pip install opencv-python dlib imutils pygame mediapipe
```

Add `shape_predictor_68_face_landmarks.dat` in the `models/` folder. Also include a `music.wav` alert sound in the project root.

---

Let me know if you also want a `README.md` or demo GIF/video instructions!
