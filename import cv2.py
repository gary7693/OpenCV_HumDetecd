import cv2
import paho.mqtt.client as mqtt
import time
import torch
from PIL import Image

# 開啟攝像頭
# 設定攝影機 ID
camera_id = 0

# 設定解析度
width = 640
height = 480

# 建立攝影機物件
cap = cv2.VideoCapture(camera_id)

# 設定解析度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 創建行人檢測器
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 捕獲第一幀
#ret, frame1 = cap.read()

while(True):
    # 捕獲每一幀
    ret, frame = cap.read()
    
    # 計算當前幀和上一幀的差異
    #diff = cv2.absdiff(frame1, frame)
    
    # 二值化處理
    #gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
     # 行人檢測
    (rects, weights) = hog.detectMultiScale(frame , winStride=(4, 4), padding=(8, 8), scale=1.05)


    # 繪製行人方框
    for (x, y, w, h) in rects:
        cv2.rectangle(frame , (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 顯示當前幀
    cv2.imshow('Webcam',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   # frame1 = frame
    time.sleep(0.01)

# 釋放攝像頭
cap.release()

# 關閉所有視窗
cv2.destroyAllWindows()