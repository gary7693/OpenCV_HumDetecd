import cv2

def display_rtsp_stream(rtsp_url):
    # 建立VideoCapture物件，並連接到RTSP串流
    cap = cv2.VideoCapture(rtsp_url)

    # 檢查是否成功連接到串流
    if not cap.isOpened():
        print("無法連接到RTSP串流")
        return

    # 持續從串流中讀取並顯示影像
    while True:
        # 讀取一幀影像
        ret, frame = cap.read()

        # 檢查是否成功讀取影像
        if not ret:
            print("無法讀取影像")
            break

        # 顯示影像
        cv2.imshow('RTSP Stream', frame)

        # 按下'q'鍵結束程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 替換為您的RTSP串流URL
    rtsp_url = "rtsp://rybstudio:ryb12345@192.168.0.90"
    display_rtsp_stream(rtsp_url)