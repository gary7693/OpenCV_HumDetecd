import cv2
import paho.mqtt.client as mqtt
import time
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].round()
    return coords

def letterbox_pil(img, imgsz=640, stride=32):
    img = Image.fromarray(img)
    shape = img.size
    new_shape = [((d - 1) // stride + 1) * stride for d in shape]
    img = img.resize(new_shape, Image.BICUBIC)
    img_np = np.array(img)
    return img_np

def detect_people(frame, model, device, imgsz=640, conf_thres=0.25, iou_thres=0.45):
    img = letterbox_pil(frame, imgsz, stride=32)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0], agnostic=False)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

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
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
device = select_device('')
model = attempt_load('yolov5s.pt')
model.to(device).eval()


while(True):
    # 捕獲每一幀
    ret, frame = cap.read()
    
    if ret:    
        # 二值化處理
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
        # 行人檢測
        #(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        detect_people(frame, model, device)

        # 繪製行人方框
       # for (x, y, w, h) in rects:
       #     cv2.rectangle(frame , (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 顯示當前幀
        cv2.imshow('Webcam',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.1)

# 釋放攝像頭
cap.release()

# 關閉所有視窗
cv2.destroyAllWindows()