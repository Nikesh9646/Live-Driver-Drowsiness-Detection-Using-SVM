# 🚘 Live Driver Drowsiness Detection using SVM

This project implements a real-time driver drowsiness detection system using a live webcam feed and machine learning (SVM classifier). It detects the driver's eye and facial features, classifies the state as Drowsy or Alert, and triggers an alarm if drowsiness is detected — helping prevent road accidents caused by fatigue.

---

## 🧠 How It Works

- Captures live video from the webcam.
- Uses dlib facial landmark detection to extract features (eye aspect ratio).
- Classifies frames using a pre-trained SVM model as either:
  - 🟢 Alert
  - 🔴 Drowsy

If drowsy, an alarm sound is played to alert the driver.

---

## 🎯 Features

- Real-time face and eye detection
- Calculates Eye Aspect Ratio (EAR)
- Classifies live video frames using SVM
- Audio alert system when drowsiness is detected
- Lightweight and runs on most systems with a webcam

---

## 🧰 Tech Stack

- Python 3
- OpenCV
- Dlib
- NumPy
- Scikit-learn (SVM)

---

## 📦 Installation

```bash
git clone https://github.com/your-username/driver-drowsiness-detection-svm.git
cd driver-drowsiness-detection-svm
pip install -r requirements.txt
```

requirements.txt includes:
- opencv-python
- dlib
- imutils
- numpy
- scikit-learn
- playsound

---

## ▶️ Run the Project

1. Make sure your webcam is connected.
2. Run:
    ```bash
    python detect_drowsiness.py
    ```
3. The system will:
    - Open the webcam
    - Detect your face and eyes
    - Display the live frame with classification (Drowsy or Alert)
    - Play an alarm if you appear drowsy

---

## 📁 Folder Structure

```
driver-drowsiness-detection-svm/
├── detect_drowsiness.py        # Main script for detection
├── train_model.py              # Script to train the SVM model
├── model.pkl                   # Trained SVM model
├── utils.py                    # EAR calculation and utilities
├── alarm.wav                   # Alarm sound file
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset

The model was trained on images extracted from:
- YawDD Dataset
- Custom image samples (open and closed eyes)

**Features Used:**
- Eye Aspect Ratio (EAR)
- Distance between facial landmarks

---

## 📊 Results

| State    | EAR Threshold | Detection  |
|----------|---------------|------------|
| Alert    | > 0.25        | No alert   |
| Drowsy   | <= 0.25       | Alarm ON   |

- **Accuracy:** ~90% on test images
- **Latency:** ~15–20 FPS in real-time webcam mode

---

## 🔮 Future Improvements

- Add mouth aspect ratio (yawning detection)
- Track head pose for distraction detection
- Deploy on mobile or Raspberry Pi
- Store drowsiness logs with timestamps

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to use, modify, and share it!

---

## ⭐ Show Your Support

If you found this useful, give it a ⭐ star and share your feedback!
