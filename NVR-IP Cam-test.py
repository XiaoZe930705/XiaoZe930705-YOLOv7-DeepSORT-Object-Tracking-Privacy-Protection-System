import cv2

def main():
    # 替換為你的攝影機 RTSP 串流地址
    rtsp_url = "rtsp://admin:Mlab03+-@192.168.0.118:554/cam/realmonitor?channel=1&subtype=0"

    # 打開攝影機串流
    print(f"正在嘗試連接到攝影機: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    # 確保攝影機可以正常連接
    if not cap.isOpened():
        print("無法連接到攝影機，請檢查 RTSP URL 或網路連線")
        return

    print("成功連接攝影機！按 'q' 鍵退出程式。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面，請檢查設定")
            break

        # 顯示攝影機畫面
        cv2.imshow("Dahua Camera Stream", frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    print("已退出程式並釋放資源。")

if __name__ == "__main__":
    main()
