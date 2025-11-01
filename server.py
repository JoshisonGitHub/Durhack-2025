import os
import requests
from flask import Flask, render_template, jsonify, request

# Template folder
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=template_dir)

# Arduino's IP address
ARDUINO_IP = "10.188.91.150"  # update if different

@app.get("/")
def index():
    return render_template("button.html")

@app.route("/press", methods=["GET", "POST"])
def press():
    print("Button pressed from", request.remote_addr)
    try:
        # Send HTTP request to Arduino
        r = requests.get(f"http://{ARDUINO_IP}/shoot", timeout=2)
        print("Arduino response:", r.text)
        return jsonify(ok=True, arduino=r.text)
    except requests.exceptions.RequestException as e:
        print("Error contacting Arduino:", e)
        return jsonify(ok=False, error=str(e))

if __name__ == "__main__":
    # Run server accessible from phone/laptop on same Wi-Fi
    app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")