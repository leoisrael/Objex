from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler
from ultralytics import YOLO
import cv2, numpy as np, base64, mediapipe as mp, math, time
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

app = Flask(__name__, static_folder='../static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# --- Modelos ---
yolo = YOLO("yolov8n.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

# --- Tracker Centroid para IDs estáveis ---
class CentroidTracker:
    def __init__(self, maxDisappeared=15, maxDistance=100):
        self.nextObjectID = 0
        self.objects = {}       # ID -> box
        self.disappeared = {}   # ID -> frames desaparecendo
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, box):
        self.objects[self.nextObjectID] = box
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objID in list(self.disappeared):
                self.disappeared[objID] += 1
                if self.disappeared[objID] > self.maxDisappeared:
                    self.deregister(objID)
            return self.objects

        # Calcula centróides
        inputCentroids = []
        for (x1,y1,x2,y2) in rects:
            cX = (x1 + x2)//2
            cY = (y1 + y2)//2
            inputCentroids.append((cX,cY))

        if len(self.objects) == 0:
            for box in rects:
                self.register(box)
        else:
            objectIDs = list(self.objects.keys())
            objectBoxes = list(self.objects.values())
            objectCentroids = [
                ((b[0]+b[2])//2,(b[1]+b[3])//2)
                for b in objectBoxes
            ]
            # matriz de distâncias
            D = np.linalg.norm(
                np.array(objectCentroids)[:,None] - np.array(inputCentroids)[None,:],
                axis=2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row,col) in zip(rows,cols):
                if row in usedRows or col in usedCols: continue
                if D[row,col] > self.maxDistance: continue

                objID = objectIDs[row]
                self.objects[objID] = rects[col]
                self.disappeared[objID] = 0
                usedRows.add(row); usedCols.add(col)

            # desaparecidos
            unusedRows = set(range(D.shape[0])) - usedRows
            for row in unusedRows:
                objID = objectIDs[row]
                self.disappeared[objID] += 1
                if self.disappeared[objID] > self.maxDisappeared:
                    self.deregister(objID)

            # novos
            unusedCols = set(range(len(inputCentroids))) - usedCols
            for col in unusedCols:
                self.register(rects[col])

        return self.objects

ct = CentroidTracker()

# --- Cache de descrições e confirmação em múltiplos frames ---
# description_cache[objectID] = { "desc": str, "time": float }
description_cache = {}
# pending_cache[objectID] = { "desc": str, "count": int }
pending_cache = {}
TTL = 30.0       # seconds para expirar descrição
CONFIRM = 4      # frames para confirmar nova descrição

def euclidean_distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def generate_description(pil_img):
    inputs = processor(pil_img, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@socketio.on('frame')
def handle_frame(data):
    try:
        now = time.time()
        # expira descrições antigas
        for objID in list(description_cache):
            if now - description_cache[objID]["time"] > TTL:
                del description_cache[objID]

        # decodifica frame
        img_bytes = base64.b64decode(data.split(",")[1])
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # YOLO detecta
        yres = yolo(frame)[0]
        person_count = 0
        detected = []
        for box in yres.boxes:
            cls = int(box.cls[0]); label = yolo.names[cls]
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            if label == "person":
                person_count += 1
            else:
                detected.append({"box":[x1,y1,x2,y2], "label":label})

        # MediaPipe para punhos
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = pose.process(rgb)
        hands = []; landmarks = []; connections = []
        if mp_res.pose_landmarks:
            h,w,_ = frame.shape
            for lm in mp_res.pose_landmarks.landmark:
                landmarks.append([int(lm.x*w), int(lm.y*h)])
            for idx in (15,16):
                lm = mp_res.pose_landmarks.landmark[idx]
                hands.append((int(lm.x*w), int(lm.y*h)))
            connections = [[a,b] for a,b in POSE_CONNECTIONS]

        # filtra objetos "na mão"
        in_hand = []
        TH = 120
        for o in detected:
            bx1,by1,bx2,by2 = o["box"]
            center = ((bx1+bx2)//2,(by1+by2)//2)
            for hx,hy in hands:
                if euclidean_distance(center,(hx,hy)) < TH:
                    in_hand.append(o)
                    break

        # aplica tracker para IDs
        rects = [o["box"] for o in in_hand]
        objects = ct.update(rects)

        tracked = []
        for objID, box in objects.items():
            # encontra o correspondente em in_hand
            for o in in_hand:
                if o["box"] == box:
                    # se já tem descrição estável, usa ela
                    if objID in description_cache:
                        o["description"] = description_cache[objID]["desc"]
                    else:
                        # TTL expired ou ainda sem desc: gerencia pending
                        if objID not in pending_cache:
                            # gera descrição primeiro frame
                            x1,y1,x2,y2 = box
                            crop = frame[y1:y2, x1:x2]
                            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            desc = generate_description(pil)
                            pending_cache[objID] = {"desc":desc, "count":1}
                        else:
                            # conta frames consistentes
                            pending_cache[objID]["count"] += 1

                        # só fixa no cache após CONFIRM frames
                        if pending_cache[objID]["count"] >= CONFIRM:
                            description_cache[objID] = {
                                "desc": pending_cache[objID]["desc"],
                                "time": now
                            }
                            del pending_cache[objID]
                            o["description"] = description_cache[objID]["desc"]
                        # antes de confirmação, ainda mostra o label
                    o["id"] = objID
                    tracked.append(o)
                    break

        emit("boxes", {
            "people": person_count,
            "in_hand": tracked,
            "pose": {"landmarks":landmarks, "connections":connections}
        })

    except Exception as e:
        print("Erro ao processar frame:", e)

if __name__ == "__main__":
    print("Servindo em https://0.0.0.0:8000")
    http_server = WSGIServer(
        ("0.0.0.0", 8000), app,
        keyfile="key.pem", certfile="cert.pem",
        handler_class=WebSocketHandler
    )
    http_server.serve_forever()
