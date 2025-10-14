import os
from flask import Flask

app = Flask(__name__)

# Use RunPod's default port 80
PORT = int(os.environ.get('PORT', 80))

@app.route('/')
def hello():
    return "Hello from Runpod Serverless!"

@app.route('/ping')
def ping():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)