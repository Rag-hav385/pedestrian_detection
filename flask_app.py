from flask import Flask, render_template, Response
import cv2
import numpy as np

#global CONF_THRESH = 0.5
#global NMS_THRESH = 0.5
# Load the network
net = cv2.dnn.readNetFromDarknet("darknet/cfg/yolov3_custom.cfg", 'darknet/backup/yolov3_custom_final.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]



app = Flask(__name__)

camera = cv2.VideoCapture('TownCentreXVID.mp4')  # use 0 for web camera


def gen_frames():
    number = 0
    while True:
        number += 1
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        elif number%100 != 0:
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            #ret, img = cv2.read(img)
            height, width = img.shape[:2]

            blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), swapRB=True, crop=False)
            net.setInput(blob)
            layer_outputs = net.forward(output_layers)

            class_ids, confidences, b_boxes = [], [], []


            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]


                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    temp = [x, y, int(w), int(h)]

                    b_boxes.append(temp)
                    confidences.append(float(confidence))


            indices = cv2.dnn.NMSBoxes(b_boxes, confidences, 0.5, 0.5).flatten().tolist()

            for index in indices:
                x, y, w, h = b_boxes[index]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            print("No of predestrian predicted at {} frame is {}".format(number , len(indices)))


@app.route('/video_feed')
def video_feed():

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
