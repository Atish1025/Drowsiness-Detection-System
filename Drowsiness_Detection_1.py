import cv2
import numpy as np
import mediapipe as mp
from pygame import mixer

# ✅ Load alarm sound
mixer.init()
mixer.music.load("music.wav")  # Ensure this file is in your project folder

# ✅ Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ✅ Use phone camera (Camo Studio). Try 1 or 0 depending on system.
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Could not open camera. Try index 0 or 2.")
    exit()

# ✅ Eye landmark indices (6-point eye region)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ✅ Thresholds
RADIUS_THRESHOLD = 20.0
FRAME_THRESHOLD = 20
flag = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Frame not received.")
        continue

    frame = cv2.resize(frame, (800, 600))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            # ✅ Collect eye landmark points
            for idx in LEFT_EYE_IDX:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for idx in RIGHT_EYE_IDX:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if len(left_eye) >= 3 and len(right_eye) >= 3:
                (lx, ly), l_radius = cv2.minEnclosingCircle(np.array(left_eye))
                (rx, ry), r_radius = cv2.minEnclosingCircle(np.array(right_eye))

                avg_radius = (l_radius + r_radius) / 2.0
                print(f"[DEBUG] Avg Radius: {avg_radius:.2f}, Flag: {flag}")

                # ✅ Draw eye circles
                cv2.circle(frame, (int(lx), int(ly)), int(l_radius), (255, 0, 0), 1)
                cv2.circle(frame, (int(rx), int(ry)), int(r_radius), (255, 0, 0), 1)

                # ✅ Trigger alarm if radius is small
                if avg_radius < RADIUS_THRESHOLD:
                    flag += 1
                    if flag >= FRAME_THRESHOLD:
                        cv2.putText(frame, "DROWSINESS ALERT!", (150, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        if not mixer.music.get_busy():
                            mixer.music.play()
                else:
                    flag = 0
                    mixer.music.stop()

    else:
        print("❌ No face detected")

    # ✅ Show running status
    cv2.putText(frame, "Press 'Q' to quit", (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
