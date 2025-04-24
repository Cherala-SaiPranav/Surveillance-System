import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
import mysql.connector
from datetime import datetime

# MySQL setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="surveillance_sys"
)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event TEXT,
    timestamp DATETIME,
    location VARCHAR(255)
)
""")

# Mediapipe setup
face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# YOLO setup
net = cv2.dnn.readNet("models/yolov4.weights", "models/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Flask setup
app = Flask(__name__)

# Event logger
def log_event(event):
    current_time = datetime.now()

    cursor.execute("SELECT MAX(timestamp) FROM events WHERE event= %s", (event,))
    last_event_time = cursor.fetchone()[0]

    if last_event_time is None or (current_time - last_event_time).total_seconds() > 60:
        event_data = (
            event,
            datetime.now(),
            "Camera 1"
        )
        cursor.execute("INSERT INTO events (event, timestamp, location) VALUES (%s, %s, %s)", event_data)
        conn.commit()

# object detection
def detect_obj(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[3] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y +h), (255, 0, 0), 2)
                log_event(f"Object detected: Class ID {class_id} with confidence (confidence: .2f)")

# face detection
def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            log_event("Face Detected")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot Capture Video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot capture video.")
            break

        detect_faces(frame)
        detect_obj(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(main(), mimetype ='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)