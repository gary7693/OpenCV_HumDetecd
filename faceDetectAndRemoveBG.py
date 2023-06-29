import cv2
import numpy as np
import time

# 載入人臉分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 初始化Webcam
cap = cv2.VideoCapture(0)

while True:
    # 讀取Webcam畫面
    ret, frame = cap.read()

    # 轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 檢測人臉
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 繪製人臉區域
    for (x, y, w, h) in faces:
        # 抽離背景
        if w<=400 or h<=400 :
            continue
            
        face = frame[y:y+h, x:x+w]
        mask = np.zeros(face.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (1, 1, w-2, h-2)
        cv2.grabCut(face, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        face_no_bg = face * mask2[:, :, np.newaxis]
        
        cv2.imshow('Face', face_no_bg)
        
        # 輸出人臉影像為JPG檔案，檔名包含時間戳        
        cv2.imwrite(f'images/detected_face_{time.time()}.jpg', face_no_bg)

        # 繪製人臉
        frame[y:y+h, x:x+w] = face_no_bg

    # 顯示結果
    cv2.imshow('Face Detection', frame)

    # 按下'q'鍵結束迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()