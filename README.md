Face, Glasses, and Hat Detection System
This project is a web-based application that detects faces, glasses, and hats in uploaded videos and images. It uses a Flask backend for processing and a Next.js frontend for displaying the results.

Features
Face Detection: Detects faces in videos and images and matches them with a known image.
Glasses Detection: Identifies the presence of glasses (clear lens glasses or sunglasses) in videos.
Hat Detection: Detects hats in videos.
Real-Time Feedback: Provides real-time updates using WebSocket (Socket.IO).
Technology Stack
Backend:

Flask: Python-based micro web framework.
Flask-CORS: For handling Cross-Origin Resource Sharing (CORS).
Flask-SocketIO: Enables real-time communication between the server and clients.
OpenCV: Library for computer vision tasks.
face_recognition library: Python library for face detection and recognition.
YOLO (You Only Look Once): Object detection model for detecting glasses and hats.
Frontend:

The frontend repository can be found [here](https://github.com/Thitiyaproud/senior-project-front-end.git)
Next.js: React framework for server-side rendering.
Tailwind CSS (optional): For styling.
Prerequisites
Python 3.7 or higher
Node.js and npm (for running the Next.js frontend)
Required Python libraries: Flask, Flask-CORS, Flask-SocketIO, OpenCV, face_recognition, ultralytics

## License
This project is licensed under the MIT License - see the LICENSE file for details.