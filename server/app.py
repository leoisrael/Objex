from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import mediapipe as mp
import math

app = Flask(__name__, static_folder='../static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

model = YOLO("yolov8n.pt")
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@socketio.on('frame')
def handle_frame(data):
    try:
        image_data = data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        yolo_results = model(frame)[0]
        detected_objects = []
        person_count = 0

        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            if label == "person":
                person_count += 1
            else:
                center = [(x1 + x2) // 2, (y1 + y2) // 2]
                detected_objects.append({'box': [x1, y1, x2, y2], 'label': label, 'center': center})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        hands = []
        pose_landmarks = []
        pose_connections = []

        if results.pose_landmarks:
            h, w, _ = frame.shape
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                pose_landmarks.append([x, y])

            for idx in [15, 16]:  # left_wrist, right_wrist
                lm = results.pose_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                hands.append((x, y))

            pose_connections = [[a, b] for a, b in POSE_CONNECTIONS]

        in_hand = []
        threshold = 80
        for obj in detected_objects:
            for hand_pos in hands:
                if euclidean_distance(obj['center'], hand_pos) < threshold:
                    in_hand.append(obj)
                    break

        emit('boxes', {
            'people': person_count,
            'in_hand': in_hand,
            'pose': {
                'landmarks': pose_landmarks,
                'connections': pose_connections
            }
        })

    except Exception as e:
        print("Erro ao processar frame:", e)

if __name__ == '__main__':
    print("[*] Servindo com HTTPS e WebSocket via Gevent...")
    http_server = WSGIServer(
        ('0.0.0.0', 8000), app,
        keyfile='key.pem',
        certfile='cert.pem',
        handler_class=WebSocketHandler
    )
    http_server.serve_forever()
