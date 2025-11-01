import os
from flask import Flask, render_template, jsonify, request

# Dynamically set template path based on where THIS file lives
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")

app = Flask(__name__, template_folder=template_dir)

@app.get("/")
def index():
    return render_template("button.html")

@app.route("/press", methods=["GET", "POST"])
def press():
    print("Button pressed!", "from", request.remote_addr)
    return jsonify(ok=True)

if __name__ == "__main__":
    # server.py (only the run line changes)
    app.run(host="0.0.0.0", port=5000, ssl_context="adhoc")



