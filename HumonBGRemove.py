import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 設定背景移除器
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換成灰階圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 移除背景
    fgmask = fgbg.apply(gray)

    # 進行膨脹和侵蝕操作
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # 尋找輪廓
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 繪製輪廓
    for contour in contours:
        if cv2.contourArea(contour) < 200:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 將背景以黑色覆蓋
        frame[fgmask == 0] = 0

    # 顯示影像
    cv2.imshow('frame', frame)

    # 按下'q'鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()