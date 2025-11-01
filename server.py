from flask import Flask, request, jsonify
app = Flask(__name__)

@app.get("/")  # optional: a simple landing page
def index():
    return "Server alive. Use /press"

@app.route("/press", methods=["GET","POST"])
def press():
    print("Button pressed!", "method=", request.method, "from", request.remote_addr)
    # TODO: trigger whatever you need here
    return jsonify(ok=True)

app.run(host="0.0.0.0", port=5000)
