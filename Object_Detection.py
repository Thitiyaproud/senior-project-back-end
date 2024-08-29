import logging
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import shutil

app = Flask(__name__)
CORS(app)

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)

# โหลดโมเดลทั้งสอง
glasses_model_path = 'best (glasses160).onnx'
hat_model_path = 'best (hat106).onnx'
glasses_model = YOLO(glasses_model_path, task='detect')
hat_model = YOLO(hat_model_path, task='detect')

# Mapping class สำหรับ glasses model
glasses_class_map = {
    "Clear Lens Glasses": "glasses",
    "Sunglasses": "sunglasses",
}

@app.route('/result_glasses', methods=['POST'])
def process_glasses_video():
    logging.debug(f"Request Form Data: {request.form}")
    logging.debug(f"Request Files: {request.files}")
    
    if 'video' not in request.files or 'glasses_type' not in request.form:
        logging.error("Missing video file or glasses_type in the request")
        return jsonify({"error": "Missing video file or glasses_type in the request"}), 400
    
    return process_video(request, glasses_model, glasses_class_map)

@app.route('/result_hats', methods=['POST'])
def process_hats_video():
    logging.debug(f"Request Form Data: {request.form}")
    logging.debug(f"Request Files: {request.files}")
    
    if 'video' not in request.files or 'hat_type' not in request.form:
        logging.error("Missing video file or hat_type in the request")
        return jsonify({"error": "Missing video file or hat_type in the request"}), 400
    
    return process_video(request, hat_model)

def process_video(request, model, class_map=None):
    video = request.files['video']
    object_type = request.form.get('glasses_type', '').lower() or request.form.get('hat_type', '').lower()

    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # สร้างโฟลเดอร์ temp ถ้าไม่มี
        logging.info(f"Created temp directory at {temp_dir}")

    output_dir = os.path.join(app.root_path, "output_frames")
    
    # ลบโฟลเดอร์ output_frames ถ้ามีอยู่ก่อนแล้ว
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logging.info(f"Removed existing output_frames directory at {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output_frames directory at {output_dir}")

    video_path = os.path.join(temp_dir, video.filename)
    video.save(video_path)  # บันทึกไฟล์วิดีโอไปยัง temp_dir

    try:
        results = test_video_processing(video_path, object_type, model, class_map)
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(video_path)
        shutil.rmtree(temp_dir, ignore_errors=True)  # ลบโฟลเดอร์ชั่วคราว

    return jsonify({"status": "success", "results": results})

def test_video_processing(video_path, object_type, model, class_map=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video.")
    
    logging.info(f"Video {video_path} opened successfully.")
    
    results_list = []
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # ดึงค่า FPS และแปลงเป็นจำนวนเต็ม
    output_dir = os.path.join(app.root_path, "output_frames")
    
    # ตรวจสอบว่ามีโฟลเดอร์ output_frames อยู่หรือไม่
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output_frames directory at {output_dir}")

    detection_interval = fps  # ตั้งค่า interval ให้ตรวจจับทุกๆ X เฟรม โดย X คือค่า fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.info("End of video or failed to read frame.")
            break

        if frame_count % detection_interval == 0:  # ตรวจจับตาม interval ที่กำหนด
            # ดำเนินการตรวจจับวัตถุ
            results = model(frame, conf=0.3, iou=0.3)
            if results is not None:
                for r in results:
                    for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                        detected_class = model.names[int(cls)].lower()
                        logging.info(f"Detected class: {detected_class}, confidence: {conf}")

                        if conf < 0.3:
                            logging.info(f"Skipping detection with confidence {conf}")
                            continue

                        # ตรวจสอบว่า detected_class ตรงกับ class_map หรือไม่
                        if class_map:
                            if detected_class not in class_map.values():
                                logging.info(f"Detected class '{detected_class}' not in class_map, skipping frame")
                                continue

                        x1, y1, x2, y2 = map(int, box)
                        timestamp = round(frame_count / fps, 2)

                        detected_filename = f"{detected_class}_frame_{frame_count}.jpg"
                        detected_path = os.path.join(output_dir, detected_filename)
                        
                        # วาดกรอบรอบวัตถุที่ตรวจจับได้ด้วยสีแดงและเส้นหนา 5 pixel
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        
                        # บันทึกภาพที่มีการตีกรอบลงในโฟลเดอร์
                        success = cv2.imwrite(detected_path, frame)
                        if success:
                            logging.info(f"Successfully saved detected frame {frame_count} as {detected_path}")
                        else:
                            logging.error(f"Failed to save detected frame {frame_count} as {detected_path}")

                        with open(detected_path, "rb") as image_file:
                            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

                        results_list.append({
                            "timestamp": timestamp,
                            "class": detected_class,  
                            "confidence": float(conf),
                            "image": encoded_image,
                            "filename": detected_filename
                        })

        frame_count += 1

    cap.release()
    return results_list

@app.route('/output_frames/<filename>')
def output_frames(filename):
    return send_from_directory(os.path.join(app.root_path, 'output_frames'), filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5000, debug=True)