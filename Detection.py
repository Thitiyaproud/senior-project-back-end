import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import os
import shutil
import face_recognition
from PIL import Image, ImageDraw

# กำหนดค่าต่าง ๆ
UPLOAD_FOLDER = 'uploads'
OUTPUT_FRAMES_FOLDER = 'output_frames'
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}

# สร้างแอป Flask และกำหนดค่า CORS และ SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # ใช้ SocketIO สำหรับ WebSocket

# ฟังก์ชันสำหรับการลบและสร้างโฟลเดอร์ใหม่
def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):  # ตรวจสอบว่าโฟลเดอร์มีอยู่หรือไม่
        shutil.rmtree(folder_path)  # ลบโฟลเดอร์ที่มีอยู่
        logging.info(f"Removed existing folder at {folder_path}")  # บันทึกการลบโฟลเดอร์
    os.makedirs(folder_path, exist_ok=True)  # สร้างโฟลเดอร์ใหม่
    logging.info(f"Created new folder at {folder_path}")  # บันทึกการสร้างโฟลเดอร์ใหม่

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)

# ลบโฟลเดอร์ที่มีอยู่และสร้างใหม่สำหรับอัพโหลดและผลลัพธ์
clear_and_create_folder(UPLOAD_FOLDER)
clear_and_create_folder(OUTPUT_FRAMES_FOLDER)

# โหลดโมเดลทั้งสองสำหรับการตรวจจับแว่นและหมวก
glasses_model_path = 'best (glasses160).onnx'
hat_model_path = 'best (hat152).onnx'
glasses_model = YOLO(glasses_model_path, task='detect')
hat_model = YOLO(hat_model_path, task='detect')

# Mapping class สำหรับ glasses model
glasses_class_map = {
    "Clear Lens Glasses": "glasses",
    "Sunglasses": "sunglasses",
}

@app.route('/result_face', methods=['POST'])
def process_face():
    # ฟังก์ชันสำหรับประมวลผลการตรวจจับใบหน้า
    logging.info("Received request for face detection.")  # บันทึกการรับคำขอ
    if 'image' not in request.files or 'video' not in request.files:
        # ตรวจสอบว่ามีไฟล์ทั้งภาพและวิดีโอหรือไม่
        logging.error("Both image and video files are required.")
        return jsonify({"error": "Please provide both image and video files."}), 400

    image_file = request.files['image']
    video_file = request.files['video']

    clear_and_create_folder(UPLOAD_FOLDER)  # ลบและสร้างโฟลเดอร์สำหรับการอัปโหลดใหม่
    clear_and_create_folder(OUTPUT_FRAMES_FOLDER)  # ลบและสร้างโฟลเดอร์สำหรับเฟรมผลลัพธ์ใหม่

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)

    image_file.save(image_path)  # บันทึกไฟล์ภาพ
    video_file.save(video_path)  # บันทึกไฟล์วิดีโอ
    logging.info(f"Saved uploaded files: {image_path}, {video_path}")  # บันทึกการบันทึกไฟล์

    image_name = os.path.splitext(image_file.filename)[0]
    logging.info(f"Extracted name from uploaded image: {image_name}")  # บันทึกชื่อไฟล์ภาพที่ดึงออกมา

    known_image = face_recognition.load_image_file(image_path)  # โหลดภาพที่รู้จัก
    known_encodings = face_recognition.face_encodings(known_image)  # สร้างการเข้ารหัสใบหน้า

    if len(known_encodings) == 0:
        # ตรวจสอบว่าพบใบหน้าในภาพที่รู้จักหรือไม่
        logging.error("No faces found in the known image.")
        return jsonify({"error": "No faces found in the known image."}), 400

    known_encoding = known_encodings[0]
    logging.info("Known face encoding generated.")  # บันทึกการสร้างการเข้ารหัสใบหน้า

    video_capture = cv2.VideoCapture(video_path)  # เปิดวิดีโอสำหรับการประมวลผล

    if not video_capture.isOpened():
        # ตรวจสอบว่าวิดีโอเปิดได้หรือไม่
        logging.error("Could not open video.")
        return jsonify({"error": "Could not open video."}), 500

    logging.info("Video opened successfully.")  # บันทึกการเปิดวิดีโอสำเร็จ

    fps = video_capture.get(cv2.CAP_PROP_FPS)  # ดึงค่า FPS จากวิดีโอ
    detect_every_x_frames = int(fps)  # ตั้งค่าจำนวนเฟรมที่ใช้ในการตรวจจับ
    frame_count = 0
    detected_faces = []  # รายการสำหรับจัดเก็บใบหน้าที่ตรวจจับได้

    while True:
        ret, frame = video_capture.read()  # อ่านเฟรมจากวิดีโอ

        if not ret:
            # ตรวจสอบว่าอ่านเฟรมสำเร็จหรือไม่
            logging.info("No more frames to read, or error occurred.")
            break

        if frame_count % detect_every_x_frames == 0:
            # ตรวจจับใบหน้าทุก ๆ x เฟรม
            logging.info(f"Processing frame {frame_count} for face detection.")
            rgb_frame = frame[:, :, ::-1]  # แปลงเป็นรูปแบบสี RGB
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)  # ย่อขนาดเฟรม
            face_locations = face_recognition.face_locations(small_frame)  # ค้นหาตำแหน่งใบหน้า
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)  # สร้างการเข้ารหัสใบหน้า

            if face_locations:
                # บันทึกจำนวนใบหน้าที่ตรวจจับได้
                logging.info(f"Detected {len(face_locations)} face(s) in frame {frame_count}.")

            pil_image = Image.fromarray(rgb_frame)  # แปลงเฟรมเป็นรูปแบบภาพ PIL
            draw = ImageDraw.Draw(pil_image)  # วาดบนภาพ
            face_found = False  # ตัวแปรสำหรับตรวจสอบว่าพบใบหน้าหรือไม่

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # เปรียบเทียบใบหน้าที่พบกับใบหน้าที่รู้จัก
                matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.4)
                if matches[0]:
                    # หากพบใบหน้าที่ตรงกัน ให้วาดกรอบสีแดงรอบใบหน้า
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
                    face_found = True

            if face_found:
                
                # กำหนดค่า timestamp
                timestamp = frame_count / fps

                # บันทึกเฟรมที่ตรวจจับได้ลงในโฟลเดอร์ผลลัพธ์
                result_frame_path = os.path.join(OUTPUT_FRAMES_FOLDER, f'{image_name}_{frame_count}_{int(timestamp)}.jpg')
                pil_image.save(result_frame_path)
                logging.info(f"Saved detected face result to {result_frame_path}.")
                
                detected_faces.append({
                    "url": f"http://localhost:5000/output_frames/{os.path.basename(result_frame_path)}",
                    "name": image_name,
                    "timestamp": round(timestamp, 2)
                })

        frame_count += 1

    video_capture.release()  # ปิดการจับภาพวิดีโอ

    if not detected_faces:
        # หากไม่พบใบหน้าใด ๆ ในวิดีโอ
        logging.warning("No faces detected in the video.")
        return jsonify({"message": "No faces detected in the video.", "personName": image_name}), 200

    # ส่งผลลัพธ์ใบหน้าที่ตรวจจับได้กลับ
    return jsonify({"message": "Processing completed.", "detected_faces": detected_faces, "personName": image_name}), 200

@app.route('/result_glasses', methods=['POST'])
def process_glasses_video():
    # ฟังก์ชันสำหรับประมวลผลการตรวจจับแว่นในวิดีโอ
    logging.debug(f"Request Files: {request.files}")

    if 'video' not in request.files:
        # ตรวจสอบว่ามีไฟล์วิดีโอหรือไม่
        logging.error("Missing video file in the request")
        return jsonify({"error": "Missing video file in the request"}), 400

    return process_video(request, glasses_model, glasses_class_map)  # เรียกใช้ฟังก์ชันสำหรับประมวลผลวิดีโอ

@app.route('/result_hats', methods=['POST'])
def process_hats_video():
    # ฟังก์ชันสำหรับประมวลผลการตรวจจับหมวกในวิดีโอ
    logging.debug(f"Request Files: {request.files}")

    if 'video' not in request.files:
        # ตรวจสอบว่ามีไฟล์วิดีโอหรือไม่
        logging.error("Missing video file in the request")
        return jsonify({"error": "Missing video file in the request"}), 400

    return process_video(request, hat_model)  # เรียกใช้ฟังก์ชันสำหรับประมวลผลวิดีโอ

def process_video(request, model, class_map=None):
    # ฟังก์ชันสำหรับประมวลผลวิดีโอทั่วไป
    video = request.files['video']

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)  # บันทึกวิดีโอ
    logging.info(f"Saved video to {video_path}")

    output_dir = OUTPUT_FRAMES_FOLDER

    if os.path.exists(output_dir):
        # ลบโฟลเดอร์ที่มีอยู่และสร้างใหม่
        shutil.rmtree(output_dir)
        logging.info(f"Removed existing output_frames directory at {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output_frames directory at {output_dir}")

    try:
        results = test_video_processing(video_path, model, class_map)  # ประมวลผลวิดีโอ
    except Exception as e:
        # จัดการข้อผิดพลาดที่เกิดขึ้นระหว่างการประมวลผล
        logging.error(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # ลบไฟล์วิดีโอชั่วคราว
        os.remove(video_path)
        logging.info(f"Removed temporary video file {video_path}")

    socketio.emit('processing_complete')  # ส่งสัญญาณไปยัง client เมื่อประมวลผลเสร็จ

    return jsonify({"status": "success", "images": results})  # ส่งผลลัพธ์การประมวลผลกลับ

def test_video_processing(video_path, model, class_map=None):
    # ฟังก์ชันทดสอบการประมวลผลวิดีโอ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # ตรวจสอบว่าสามารถเปิดวิดีโอได้หรือไม่
        raise Exception("Could not open video.")

    logging.info(f"Video {video_path} opened successfully.")

    results_list = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # ดึงค่า FPS จากวิดีโอ
    output_dir = OUTPUT_FRAMES_FOLDER

    detection_interval = int(fps)  # กำหนดช่วงการตรวจจับ

    while cap.isOpened():
        ret, frame = cap.read()  # อ่านเฟรมจากวิดีโอ
        if not ret or frame is None:
            logging.info("End of video or failed to read frame.")
            break

        if frame_count % detection_interval == 0:
            # ประมวลผลการตรวจจับในเฟรมที่กำหนด
            results = model(frame, conf=0.3, iou=0.3)
            if results is not None:
                for r in results:
                    for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                        detected_class = model.names[int(cls)].lower()

                        if conf < 0.3:
                            continue

                        if class_map and detected_class not in class_map.values():
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        timestamp = frame_count / fps  # คำนวณเวลาในวิดีโอ

                        detected_filename = f"{detected_class}_frame_{int(timestamp)}.jpg"
                        detected_path = os.path.join(output_dir, detected_filename)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # วาดกรอบสีแดงรอบใบหน้าที่ตรวจจับได้
                        cv2.imwrite(detected_path, frame)  # บันทึกเฟรมที่ตรวจจับได้

                        result = {
                            "timestamp": round(timestamp, 2),
                            "class": detected_class,
                            "confidence": float(conf),
                            "filename": detected_filename,
                            "url": f"http://localhost:5000/output_frames/{detected_filename}"
                        }

                        socketio.emit('detection', result)  # ส่งสัญญาณการตรวจจับไปยัง client
                        results_list.append(result)

        frame_count += 1

    cap.release()  # ปิดการจับภาพวิดีโอ
    return results_list  # ส่งรายการผลลัพธ์กลับ

@app.route('/output_frames/<filename>')
def output_frames(filename):
    # ฟังก์ชันสำหรับส่งไฟล์เฟรมที่ตรวจจับได้
    return send_from_directory(os.path.join(app.root_path, 'output_frames'), filename, mimetype='image/jpeg')

if __name__ == '__main__':
    # รันแอป Flask ด้วย SocketIO
    socketio.run(app, port=5000, debug=True)
