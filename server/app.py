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
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__, static_folder='../static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# YOLOv8 pré-treinado em COCO
yolo = YOLO("yolov8n.pt")

# BLIP local
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# MediaPipe Pose
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

# Cache de descrições
description_cache = {}

def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def generate_description(pil_image):
    inputs = processor(pil_image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@socketio.on('frame')
def handle_frame(data):
    try:
        # Decodifica frame
        img_data = data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # YOLO detecta objetos
        res = yolo(frame)[0]
        person_count = 0
        detected = []
        for box in res.boxes:
            cls = int(box.cls[0])
            label = yolo.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == "person":
                person_count += 1
            else:
                center = [(x1 + x2)//2, (y1 + y2)//2]
                detected.append({
                    "box": [x1, y1, x2, y2],
                    "label": label,
                    "center": center
                })

        # MediaPipe para esqueleto e punhos
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = pose.process(rgb)
        landmarks, connections, hands = [], [], []
        if mp_res.pose_landmarks:
            h, w, _ = frame.shape
            for lm in mp_res.pose_landmarks.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h)])
            for idx in (15, 16):
                lm = mp_res.pose_landmarks.landmark[idx]
                hands.append((int(lm.x * w), int(lm.y * h)))
            connections = [[a, b] for a, b in POSE_CONNECTIONS]

        # Filtra objetos “na mão” por distância
        in_hand = []
        TH = 120
        for obj in detected:
            for hx, hy in hands:
                if euclidean_distance(obj["center"], (hx, hy)) < TH:
                    # Crop ampliado em 20%
                    x1, y1, x2, y2 = obj["box"]
                    w_box = x2 - x1
                    h_box = y2 - y1
                    pad_w = int(w_box * 0.2)
                    pad_h = int(h_box * 0.2)
                    # calcula coordenadas expandidas e mantém dentro da imagem
                    nx1 = max(0, x1 - pad_w)
                    ny1 = max(0, y1 - pad_h)
                    nx2 = min(frame.shape[1], x2 + pad_w)
                    ny2 = min(frame.shape[0], y2 + pad_h)
                    crop = frame[ny1:ny2, nx1:nx2]

                    # gera chave para cache
                    key = f"{obj['label']}_{nx1//50}_{ny1//50}"
                    if key not in description_cache:
                        # converte crop para PIL e gera descrição
                        from PIL import Image
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        desc = generate_description(pil_img)
                        description_cache[key] = desc
                    obj["description"] = description_cache[key]

                    in_hand.append(obj)
                    break

        # Emite pro front
        emit("boxes", {
            "people": person_count,
            "in_hand": in_hand,
            "pose": {
                "landmarks": landmarks,
                "connections": connections
            }
        })

    except Exception as e:
        print("Erro ao processar frame:", e)

if __name__ == "__main__":
    print("Servindo em https://0.0.0.0:8000")
    server = WSGIServer(
        ("0.0.0.0", 8000), app,
        keyfile="key.pem", certfile="cert.pem",
        handler_class=WebSocketHandler
    )
    server.serve_forever()
