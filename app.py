from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import cv2  # Import OpenCV

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize HandDetector
detector = HandDetector(maxHands=1, detectionCon=0.8)


@socketio.on("connect")
def test_connect():
    print("Client connected")


def detect_hands(image_data):
    # Convert image data to an OpenCV image
    np_img = np.array(Image.open(BytesIO(image_data)))

    # Convert the image from RGB to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Detect hands in the image
    hands, _ = detector.findHands(img_bgr, flipType=False)
    # print(hands)
    keypoints = []
    if hands:
        for hand in hands:
            bbox = hand["bbox"]  # Get bounding box for hand
            hand_width, hand_height = bbox[2], bbox[3]  # Extract hand dimensions
            lmList = hand["lmList"]  # List of hand landmarks
            for idx, lm in enumerate(lmList):
                keypoints.append(
                    {
                        "id": idx,
                        "x": lm[0],  # X-coordinate
                        "y": lm[1],  # Y-coordinate
                        "z": lm[2],  # Z-coordinate
                        "hand_width": hand_width,
                        "hand_height": hand_height,
                    }
                )
    return keypoints


@socketio.on("send_frame")
def handle_frame(frame_data):
    try:
        # Decode the base64 image data
        img_data = base64.b64decode(frame_data.split(",")[1])

        # Detect hands from the image
        keypoints = detect_hands(img_data)

        # Emit the detected keypoints to the frontend
        emit("hand_keypoints", {"keypoints": keypoints})

    except Exception as e:
        emit("error", {"error": str(e)})


@app.route("/")
def index():
    return "Hand Detection WebSocket Server"


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
