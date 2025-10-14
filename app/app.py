import os
from flask import Flask
from threading import Thread

app = Flask(__name__)

# Get ports from environment variables
PORT = int(os.environ.get('PORT', 80))
PORT_HEALTH = int(os.environ.get('PORT_HEALTH', 80))


@app.route('/')
def hello():
    return "Hello from Runpod Serverless!"


@app.route('/test')
def test():
    return "This is a test page!"


# If PORT and PORT_HEALTH are the same, use same app
if PORT == PORT_HEALTH:
    @app.route('/ping')
    def ping():
        return {"status": "healthy"}, 200


    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=PORT)
else:
    # Separate health check server
    health_app = Flask('health')


    @health_app.route('/ping')
    def ping():
        return {"status": "healthy"}, 200


    def run_health_server():
        health_app.run(host='0.0.0.0', port=PORT_HEALTH)


    if __name__ == '__main__':
        # Start health check server in background thread
        health_thread = Thread(target=run_health_server, daemon=True)
        health_thread.start()

        # Start main app
        app.run(host='0.0.0.0', port=PORT)