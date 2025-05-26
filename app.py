#Response is useful for sending the video stream to the browser.
from flask import Flask,render_template,Response,jsonify
import cv2 

import time

import numpy as np

from tensorflow.keras.models import load_model

# model=load_model("bpm_model.keras")

model2=load_model("bpm_model_v2.keras")

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
                    
                    X=np.array(X)
                    X=X.reshape(1,24)

                    predict=model2.predict(X)

                    print(predict)

                    bp=int(predict[0][0])

                    cv2.putText(frame,f"BPM: {int(predict[0][0])}",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 0), 2)


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

