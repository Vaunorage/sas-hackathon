import os
from flask import Flask

app = Flask(__name__)

# Get ports from environment variables (RunPod sets these)
PORT = int(os.environ.get('PORT', 8000))
PORT_HEALTH = int(os.environ.get('PORT_HEALTH', 8001))

@app.route('/')
def hello():
    return "Hello from Runpod Serverless!"

@app.route('/test')
def test():
    return "This is a test page!"

# Health check endpoint (required by RunPod)
@app.route('/ping')
def ping():
    return {"status": "ok"}

if __name__ == '__main__':
    # Run on the port RunPod expects
    app.run(host='0.0.0.0', port=PORT)