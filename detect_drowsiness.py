import cv2
import dlib
import numpy as np
import pickle
import imutils
from imutils import face_utils
from playsound import playsound
from utils import eye_aspect_ratio

# Load trained SVM model
with open("model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR threshold for drowsiness detection
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # Number of consecutive frames for alarm

COUNTER = 0
ALARM_ON = False

def main():
    global COUNTER, ALARM_ON

    cap = cv2.VideoCapture(0)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Extract features for SVM
            features = np.array([ear]).reshape(1, -1)
            prediction = svm_model.predict(features)[0]

            if prediction == 1:  # Drowsy
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES and not ALARM_ON:
                    ALARM_ON = True
                    playsound("alarm.wav", block=False)

                cv2.putText(frame, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            else:
                COUNTER = 0
                ALARM_ON = False
                cv2.putText(frame, "ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

            # Draw eye contours
            for eye in [leftEye, rightEye]:
                hull = cv2.convexHull(eye)
                cv2.drawContours(frame, [hull], -1, (255, 255, 0), 1)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
