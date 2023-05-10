import cv2
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

if __name__ == "__main__":
    # 替換為您的RTSP串流URL
    rtsp_url = "rtsp://rybstudio:ryb12345@192.168.0.90:554/stream1"
    #display_rtsp_stream(rtsp_url)
    device = select_device('')
    model = attempt_load('yolov5s.pt')
    model.to(device).eval()
    cap = cv2.VideoCapture(rtsp_url)

    # 檢查是否成功連接到串流
    if not cap.isOpened():
        print("無法連接到RTSP串流")
        

    # 持續從串流中讀取並顯示影像
    while True:
        # 讀取一幀影像
        ret, frame = cap.read()

        # 檢查是否成功讀取影像
        if not ret:
            print("無法讀取影像")
            break

        detect_people(frame, model, device)

        # 顯示影像
        cv2.imshow('RTSP Stream', frame)

        # 按下'q'鍵結束程式
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.05)

    # 釋放資源並關閉視窗
    cap.release()
    cv2.destroyAllWindows()