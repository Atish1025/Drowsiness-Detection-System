from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
from pygrabber.dshow_graph import FilterGraph
import imutils
import dlib
import cv2

# ✅ List and select camera using pygrabber
def select_camera():
    print("\n🔍 Available camera devices:")
    graph = FilterGraph()
    devices = graph.get_input_devices()
    for i, name in enumerate(devices):
        print(f"{i}: {name}")
    try:
        index = int(input("Enter camera index from above list: "))
        if 0 <= index < len(devices):
            print(f"✅ Selected: {devices[index]}")
            return index
        else:
            print("❌ Invalid index. Using default index 0.")
            return 0
    except ValueError:
        print("❌ Invalid input. Using default index 0.")
        return 0

# ✅ Load alert sound
mixer.init()
mixer.music.load("music.wav")

# ✅ Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)

# ✅ Parameters
thresh = 0.25
frame_check = 20
flag = 0

# ✅ Dlib detector and predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# ✅ Eye indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# ✅ Camera selection
camera_index = select_camera()
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
	print("❌ Camera could not be opened.")
	exit()

cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# ✅ Start detection loop
while True:
	ret, frame = cap.read()
	if not ret:
		print("❌ No frame received.")
		break

	frame = imutils.resize(frame, width=800)
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
		cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

		# Drowsiness logic
		if ear < thresh:
			flag += 1
			print(f"[DEBUG] EAR: {ear:.2f}, Count: {flag}")
			if flag >= frame_check:
				cv2.putText(frame, "******** ALERT: DROWSINESS ********", (50, 60),
							cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
				if not mixer.music.get_busy():
					mixer.music.play()
		else:
			flag = 0
			mixer.music.stop()

	# Show frame
	cv2.putText(frame, "Press 'Q' to Quit", (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
	cv2.imshow("Frame", frame)

	# Exit on 'q'
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
