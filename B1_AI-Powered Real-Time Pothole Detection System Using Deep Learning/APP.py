from flask import Flask, render_template, request, redirect, url_for, Response, session
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = "pothole_secret"


net = cv2.dnn.readNet("pothole-yolov3-tiny.weights", "pothole-yolov3-tiny.cfg")
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


users = {}   # { 'username': {'phone':..., 'password':...} }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        password = request.form["password"]

        users[name] = {"phone": phone, "password": password}
        return redirect(url_for("signin"))

    return render_template("signup.html")



@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        name = request.form["name"]
        password = request.form["password"]

        if name in users and users[name]["password"] == password:
            session["user"] = name
            return redirect(url_for("stream"))
        else:
            return redirect(url_for("signup"))

    return render_template("signin.html")



streaming = False


@app.route("/stream")
def stream():
    if "user" not in session:
        return redirect(url_for("signin"))
    return render_template("stream.html")


def gen_frames():
    global streaming

    cap = cv2.VideoCapture(0)

    while streaming:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 250, 0), 2)
                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route("/start_stream")
def start_stream():
    global streaming
    streaming = True
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/stop_stream")
def stop_stream():
    global streaming
    streaming = False
    return render_template("stream.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run()
