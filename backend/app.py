from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = load_model("bpm_model_v2.keras")
classifier = cv2.CascadeClassifier("haar_cascade_frontal_face.xml")

X = []
session_data = []  # Store BPM values for current session
face_detected = False
face_counter = 0
no_face_counter = 0

FACE_THRESHOLD = 5
NO_FACE_THRESHOLD = 10

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('frame')
def handle_frame(data):
    global X, face_detected, face_counter, no_face_counter, session_data

    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(face_rects) > 0:
        no_face_counter = 0  # reset no-face counter

        if not face_detected:
            face_counter += 1
            if face_counter >= FACE_THRESHOLD:
                face_detected = True
                face_counter = 0
                session_data = []  # Reset session data for new user
                emit('status', {'state': 'calibrating'})
            else:
                emit('status', {'state': 'noface'})
                emit('bpm', {'bpm': 0})  # send 0 until calibration starts
            return
        else:
            face_counter = 0  # reset face counter if already detected

        # Process frame
        x, y, w, h = face_rects[0]
        green = frame[y:y+h, x:x+w, 1]
        mean_g = np.mean(green)
        X.append(mean_g)

        if len(X) == 24:
            X_input = np.array(X).reshape(1, 24)
            prediction = model.predict(X_input)[0][0]
            bpm = int(prediction)
            session_data.append(bpm)  # Add to session data
            emit('bpm', {'bpm': bpm})
            emit('status', {'state': 'live', 'bpm': bpm})
            X = []

    else:
        face_counter = 0  
        no_face_counter += 1

        if no_face_counter >= NO_FACE_THRESHOLD:
            if face_detected:
                if session_data:
                    avg_bpm = sum(session_data) / len(session_data)
                    emit('session_end', {'avg_bpm': avg_bpm})
                face_detected = False
                X.clear()
                session_data.clear()
                emit('status', {'state': 'noface'})
                emit('bpm', {'bpm': 0})


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
