import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pickle
import threading

with open('model_ASL.pkl', 'rb') as f:
    model = pickle.load(f)
    print('work')

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process_camera_feed():
    global output_frame, whole_word_text, detected_letters  

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame from the camera.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            body_language_class = "None"
            body_language_prob = None

            if results.left_hand_landmarks:
                pose = results.left_hand_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                row = pose_row
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]

            key = cv2.waitKey(10)

            current_sign_text = f"Current Sign: {body_language_class}"

            cv2.putText(image, current_sign_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Video Feed", image)

            with lock:
                output_frame = image.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()

threading.Thread(target=process_camera_feed).start()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
