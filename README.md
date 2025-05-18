# Objex

Real-time augmented-reality web app that detects objects in your hand, overlays a pose skeleton, and generates AI-driven descriptive captions — all running locally with no paid APIs.

---

## 🚀 Features

- **Real-time object detection** using YOLOv8 (80 COCO classes)  
- **Pose estimation & skeleton overlay** via MediaPipe  
- **AI-driven captions** with BLIP (`Salesforce/blip-image-captioning-base`)  
- **Stable multi-frame tracking** (CentroidTracker + 4-frame confirmation)  
- **Description cache with TTL** (30 s) to avoid repeated captioning  
- **WebSocket communication** (Flask + Flask-SocketIO + gevent)  
- **Responsive web UI**: live video, bounding boxes, skeleton, preview pane  
- **Switchable & fullscreen camera** controls  

---

## 📦 Tech Stack

- **Backend**: Python 3.9+, Flask, Flask-SocketIO, gevent, ultralytics (YOLOv8), MediaPipe, OpenCV, Hugging Face Transformers (BLIP), Pillow  
- **Frontend**: HTML5, CSS3, JavaScript, Socket.IO  

---

## 📂 Project Structure

```
├── server/
│   └── app.py             # Flask + SocketIO backend
├── static/
│   └── index.html         # Web UI (video, canvas, controls)
├── cert.pem               # SSL certificate
├── key.pem                # SSL private key
└── README.md              # This file
```

---

## ⚙️ Installation

1. **Clone repository**  
   ```bash
   git clone https://github.com/yourusername/Objex.git
   cd Objex
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   _requirements.txt_ should include:
   ```
   flask
   flask-socketio
   gevent
   gevent-websocket
   ultralytics
   opencv-python
   mediapipe
   transformers
   torch
   pillow
   ```

4. **Generate SSL cert & key** (if not included)  
   ```bash
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   ```

---

## ▶️ Usage

1. **Start the server**  
   ```bash
   python server/app.py
   ```
   By default, serves at **https://0.0.0.0:8000**.

2. **Open the web UI** on your tablet or PC:  
   ```
   https://<server-ip>:8000
   ```

3. **Allow camera access** in the browser.  
   - Click **"Trocar Câmera"** to switch front/back camera.  
   - Click **"Fullscreen"** for immersive view.  

4. **Interact**: hold objects in your hand — bounding boxes, skeleton and AI captions will appear in real time.  

---

## 🎨 Screenshots

<!--
Replace `path/to/...` with your actual image paths or URLs.
-->
| Live View | Description & Preview |
|:---------:|:---------------------:|
| ![Live view with skeleton and boxes](path/to/live_view.png) | ![Preview pane with AI caption](path/to/preview.png) |
| ![Fullscreen mode demo](path/to/fullscreen.png) | ![Objects in hand list](path/to/list_view.png) |

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add awesome feature"`)  
4. Push (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

> Built with ❤️ by [Your Name]  
> Inspired by futuristic AR interfaces and powered by open-source AI models.
