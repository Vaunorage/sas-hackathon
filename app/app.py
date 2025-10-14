import os
from flask import Flask, jsonify

app = Flask(__name__)

# Get port from environment (default 80)
PORT = int(os.environ.get('PORT', 80))


@app.route('/')
def hello():
    return jsonify({"message": "Hello from Runpod Serverless!"})


@app.route('/test')
def test():
    return jsonify({"message": "This is a test page!"})


# Health check endpoint (REQUIRED by RunPod)
@app.route('/ping')
def ping():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    print("=" * 50)
    print(f"STARTING FLASK ON PORT {PORT}")
    print("=" * 50)

    # Use Werkzeug's simple server (built into Flask)
    app.run(host='0.0.0.0', port=PORT, debug=False)