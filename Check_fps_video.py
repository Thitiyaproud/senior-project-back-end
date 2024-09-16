import cv2

# ฟังก์ชันสำหรับตรวจสอบ FPS ของวิดีโอ
def check_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # ตรวจสอบ FPS ของวิดีโอ
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps}")

    cap.release()

# Path ไปยังไฟล์วิดีโอสำหรับทดสอบ
video_path = r'E:\test_3m.mp4'

# เรียกใช้งานฟังก์ชันตรวจสอบ FPS
check_video_fps(video_path)
