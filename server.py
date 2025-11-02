import os
import requests
from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
from glob import glob
import webcam


import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaBlackhole
from flask import Flask, render_template, request, jsonify
import cv2
import uuid
import logging
from av import VideoFrame
import argparse
"""
import asyncio
from flask import request
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av import VideoFrame
"""

# --- WebRTC globals ---
pcs = set()
_latest_jpeg = None
_latest_lock = threading.Lock()

# --------------------------
# Flask Setup
# --------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")
app = Flask(__name__, template_folder=template_dir)

# --------------------------
# Arduino Setup
# --------------------------
ARDUINO_IP = "10.188.91.150"  # update if different
"""
# --------------------------
# OpenCV Setup
# --------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Load enemy sprites (expect 5)
enemy_paths = sorted(glob(os.path.join(base_dir, "static", "enemies", "*.png")))
enemies = [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in enemy_paths]
if len(enemies) != 5:
    raise Exception("Expected 5 enemy sprites in /static/enemies")

# --------------------------
# HSV color ranges (camera-tuned)
# --------------------------
# Ordered: Red, Orange, Yellow, Green, Blue
color_hsv_ranges = {
    0: (np.array([0, 150, 150]), np.array([8, 255, 255])),   # Red
    1: (np.array([10, 150, 150]), np.array([22, 255, 255])), # Orange
    2: (np.array([23, 150, 150]), np.array([33, 255, 255])), # Yellow
    3: (np.array([50, 80, 80]), np.array([70, 255, 255])),   # Green
    4: (np.array([100, 80, 100]), np.array([120, 220, 255])) # Blue
}

# --------------------------
# Helper functions
# --------------------------
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    h, w = img_overlay.shape[:2]
    y1, y2 = max(0, y), min(img.shape[0], y+h)
    x1, x2 = max(0, x), min(img.shape[1], x+w)
    y1o, y2o = max(0, -y), min(h, img.shape[0]-y)
    x1o, x2o = max(0, -x), min(w, img.shape[1]-x)

    if y1 >= y2 or x1 >= x2:
        return img

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    img_crop[:] = alpha * img_overlay_crop[:, :, :3] + (1 - alpha) * img_crop
    return img

def detect_cups(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = []

    for idx, (lower, upper) in color_hsv_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 1500:  # slightly lower for green cups
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (cv2.contourArea(cnt) / (perimeter * perimeter + 1e-5))
            if circularity < 0.4:  # slightly relaxed
                continue
            detected.append({'color_idx': idx, 'bbox': (x, y, w, h)})
    return detected

def assign_enemy(frame, cup):
    x, y, w, h = cup['bbox']
    enemy = enemies[cup['color_idx']]
    enemy_resized = cv2.resize(enemy, (w, h), interpolation=cv2.INTER_AREA)
    alpha = enemy_resized[:, :, 3] if enemy_resized.shape[2] == 4 else np.ones((h, w), dtype=np.uint8) * 255
    frame = overlay_image_alpha(frame, enemy_resized, x, y, alpha)
    return frame

def process_frame(frame):
    cups = detect_cups(frame)
    for cup in cups:
        frame = assign_enemy(frame, cup)
    return frame
"""
# --------------------------
# Flask routes
# --------------------------
@app.get("/")
def index():
    return render_template("button.html")

@app.route("/press", methods=["GET", "POST"])
def press():
    try:
        r = requests.get(f"http://{ARDUINO_IP}/shoot", timeout=2)
        return jsonify(ok=True, arduino=r.text)
    except requests.exceptions.RequestException as e:
        return jsonify(ok=False, error=str(e))
"""
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""

# --------------------------
# WebRTC: receive video from browser
# --------------------------
async def _consume_video(track):
    global _latest_jpeg
    import cv2
    while True:
        frame = await track.recv()
        # Convert to JPEG bytes for optional preview endpoints
        img = frame.to_ndarray(format="bgr24")
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            with _latest_lock:
                _latest_jpeg = buf.tobytes()

@app.post("/offer")
def offer():
    # Accept a WebRTC SDP offer from the browser, create an answer, and
    # start reading the incoming video track.
    # Requires: pip install aiortc av opencv-python
    params = request.get_json(force=True, silent=False)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    async def run():
        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                # consume frames (update _latest_jpeg)
                asyncio.create_task(_consume_video(track))
            else:
                # if an audio track ever arrives, sink it
                bh = MediaBlackhole()
                await bh.start()
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    # run the async logic in this sync Flask route
    return jsonify(asyncio.run(run()))
"""
from fastapi.responses import StreamingResponse

@app.get("/signal")
async def stream_updates(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            # Output is the AdditionalOutputs instance
            # Be sure to serialize it however you would like
            yield f"data: {output.args[0]}\n\n"

    return StreamingResponse(
        output_stream(), 
        media_type="text/event-stream"
    )
"""
@app.get("/webrtc_preview")
def webrtc_preview():
    def gen():
        boundary = b"--frame\r\n"
        while True:
            with _latest_lock:
                data = _latest_jpeg
            if data is not None:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            else:
                import time; time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")






#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=5000, use_reloader=False, threaded=False)






"""
# --------------------------
# Run Flask app
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

    app2 = webcam.web.Application()
    app2.on_shutdown.append(webcam.on_shutdown)
    app2.router.add_get("/", index)
    app2.router.add_post("/offer", offer)
    webcam.web.run_app(app2, host="0.0.0.0", port=8080)
"""
