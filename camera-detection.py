import cv2

def find_available_camera():
    for i in range(10):  # 假设最多有10个相机，可以调整范围
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"相机 {i} 可用")
            cap.release()
        else:
            print(f"相机 {i} 不可用")

find_available_camera()
