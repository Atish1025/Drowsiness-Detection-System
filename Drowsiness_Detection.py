from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize sound alert
mixer.init()
mixer.music.load("music.wav")  # Ensure this file exists in your project directory

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# Threshold values
thresh = 0.25
frame_check = 20

# Dlib face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Path to model

# Get indexes for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Use phone camera via Camo Studio
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try 2 or 3 if 1 doesn't work

# Check if camera opened successfully
if not cap.isOpened():
	print("❌ Camera not detected. Try a different index (0, 1, 2, etc.)")
	exit()

# Optional: Set window to full screen
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

flag = 0
while True:
	ret, frame = cap.read()
	if not ret:
		break

	frame = imutils.resize(frame, width=800)  # Enlarged display
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0

		# Draw eye contours
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < thresh:
			flag += 1
			print("Drowsiness frame count:", flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10, 750),
							cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				mixer.music.play()
		else:
			flag = 0

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# Cleanup
cap.release()
cv2.destroyAllWindows()
