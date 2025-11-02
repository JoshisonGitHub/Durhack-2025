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


# --------------------------
# Flask Setup
# --------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")
app = Flask(__name__, template_folder=template_dir)
pcs = set()
logger = logging.getLogger("pc")

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

class VideoTransformTrack(MediaStreamTrack):
	kind = "video"
	def __init__(self, track, transform):
		super().__init__()
		self.track = track
		self.transform = transform

	async def recv(self):
		frame = await self.track.recv()

		if self.transform == "<some tranformation>":
			pass
		else:
			return frame

class OpenCVMediaStreamTrack(MediaStreamTrack):
	kind = "video"
	
	def __init__(self, device_index=0):
		super().__init__()
		self.cam = cv2.VideoCapture(device_index)

	async def recv(self):
		success, frame = self.cam.read()
		if not success:
			raise Exception("Failed to capture frames")
		
		# Convert the image from OpenCV format to AV format
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = VideoFrame.from_ndarray(frame, format="rgb24")
		frame.pts = frame.time_base = None
		return frame

@app.route('/offer', methods=['POST'])
async def offer():
	params = request.get_json() # synchronous
	if not params:
		return jsonify({"error": "Invalid JSON data"}), 400
	offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
	pc = RTCPeerConnection()
	pc_id = "PeerConnection(%s)" % uuid.uuid4()
	pcs.add(pc)

	def log_info(msg, *args):
		logger.info(pc_id + " " + msg, *args)

	log_info("Created for %s", request.remote_addr)

	# player = MediaPlayer('video=Integrated Camera', format='dshow', options={'frame_rate': "60", 'video_size': '640x480'})
	if args.record_to:
		recorder = MediaRecorder(args.record_to)
	else:
		recorder = MediaBlackhole()

	@pc.on("connectionstatechange")
	async def on_connectionstatechange():
		log_info("Connection state is %s", pc.connectionState)
		if pc.connectionState == "failed":
			await pc.close()
			pcs.discard(pc)

	@pc.on("track")
	async def on_track(track):
		log_info("Track %s received", track.kind)
		
		if track.kind == "audio":
			pc.addTrack(track)
			recorder.addTrack(track)
		elif track.kind == "video":
			# pc.addTrack(VideoTransformTrack(relay.subscribe(track)), transform=params["video_transform"])
			video_track = OpenCVMediaStreamTrack(device_index=0)
			print(video_track)
			pc.addTrack(VideoTransformTrack(video_track, transform=params.get("video_transform", "")))
			if args.record_to:
				recorder.addTrack(video_track)
		
		@track.on("ended")
		async def on_ended():
			log_info("Track %s ended", track.kind)
			await recorder.stop()

	# handle offer
	await pc.setRemoteDescription(offer)
	await recorder.start()

	# send answer
	answer = await pc.createAnswer()
	await pc.setLocalDescription(answer)

	# Handle offer
	# loop = asyncio.new_event_loop()
	# asyncio.set_event_loop(loop)
	# loop.run_until_complete(pc.setRemoteDescription(offer))
	# loop.run_until_complete(recorder.start())

	# # Send answer
	# answer = loop.run_until_complete(pc.createAnswer())
	# loop.run_until_complete(pc.setLocalDescription(answer))
	
	return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC+Flask Live Streaming Application")
    parser.add_argument("--record_to", help="Write received media to a file")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
