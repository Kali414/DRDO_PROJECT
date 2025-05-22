#Response is useful for sending the video stream to the browser.
from flask import Flask,render_template,Response
import cv2

app=Flask(__name__)
camera=cv2.VideoCapture(0)

# classifier=cv2.CascadeClassifier("haar_cascade_eye.xml")

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # eyes_react=classifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
            # for (x,y,w,h) in eyes_react:
            #     cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
            
            # The code is converting the frame to jpg format.
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()


        # This is used to send the frame to the browser.
        # It sends the frame in chunks of data.
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

