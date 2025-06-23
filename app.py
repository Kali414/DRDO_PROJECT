#Response is useful for sending the video stream to the browser.
from flask import Flask,render_template,Response,jsonify
import cv2 

import time

import numpy as np

from tensorflow.keras.models import load_model

# model=load_model("bpm_model.keras")


## both data1 and data2 model

model_list=["bpm_model_hd.keras","bpm_model_hd1.keras","bpm_model_hm.keras","bpm_model_m.keras","bpm_model_md.keras","bpm_model_reshaped.keras","best_cnn_lstm_model.keras"]

model2=load_model("best_cnn_lstm2_model.keras")

app=Flask(__name__)
camera=cv2.VideoCapture(0)

classifier=cv2.CascadeClassifier("haar_cascade_frontal_face.xml")

cnt=0
bp=0

def generate_frames():
    global cnt,bp
    X=[]
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            a=time.time()
            cnt+=1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_react=classifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
            
            
            if len(face_react) > 0:
                x, y, w, h = face_react[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                green = frame[y:y + h, x:x + w, 1]  # Green channel
                mean_g = np.mean(green)
                X.append(mean_g)

                if(len(X)==24):
                    
                    if(X!=0):
                        X=np.array(X)
                        X=X.reshape(1,24)

                        predict=model2.predict(X)

                        print(predict)

                        bp=int(predict[0][0])

                        cv2.putText(frame,f"BPM: {int(predict[0][0])}",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2)

                    else:
                        bp=0


                    X = []  # Reset for next sample

        ## Code for live display

        # The code is converting the frame to jpg format.
        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()


        # This is used to send the frame to the browser.
        # It sends the frame in chunks of data.
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.2)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/bpm")
def bpm():
    return jsonify({"bpm":bp})

if __name__=="__main__":
    app.run(debug=True)



# # Response is useful for sending the video stream to the browser.
# from flask import Flask, render_template, Response, jsonify
# import cv2
# import time
# import numpy as np
# import mediapipe as mp
# from tensorflow.keras.models import load_model


# model_list=["bpm_model_hd.keras","bpm_model_hd1.keras","bpm_model_hm.keras","bpm_model_m.keras","bpm_model_md.keras"]

# # Load the model
# model2 = load_model("bpm_model_hd.keras")

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)

# cnt = 0
# bp = 0

# # Initialize MediaPipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# def generate_frames():
#     global cnt, bp
#     X = []

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             a = time.time()
#             cnt += 1

#             # Convert frame to RGB for MediaPipe
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_detection.process(rgb)

#             if results.detections:
#                 # Get first detected face
#                 detection = results.detections[0]
#                 bbox = detection.location_data.relative_bounding_box

#                 h, w, _ = frame.shape
#                 x = int(bbox.xmin * w)
#                 y = int(bbox.ymin * h)
#                 w_box = int(bbox.width * w)
#                 h_box = int(bbox.height * h)

#                 # Clamp values to frame size
#                 x = max(0, x)
#                 y = max(0, y)
#                 x2 = min(frame.shape[1], x + w_box)
#                 y2 = min(frame.shape[0], y + h_box)

#                 # Draw rectangle
#                 cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

#                 # Extract green channel from face region
#                 green = frame[y:y2, x:x2, 1]
#                 mean_g = np.mean(green)
#                 X.append(mean_g)

#                 if len(X) == 24:
#                     X = np.array(X).reshape(1, 24)
#                     predict = model2.predict(X)
#                     bp = int(predict[0][0])

#                     # Display BPM
#                     cv2.putText(frame, f"BPM: {bp}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                     X = []

#         # Encode frame as JPEG
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         # Send to browser
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         time.sleep(0.2)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/bpm")
# def bpm():
#     return jsonify({"bpm": bp})

# if __name__ == "__main__":
#     app.run(debug=True)

