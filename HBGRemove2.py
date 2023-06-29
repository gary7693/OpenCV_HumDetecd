import cv2

# 載入模型
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# 設定類別標籤
class_labels = {0: 'background', 1: 'person'}

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 進行人物偵測
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
    model.setInput(blob)
    output = model.forward()

    # 繪製偵測框
    for detection in output[0, 0, :, :]:
        score = float(detection[2])
        class_id = int(detection[1])
        if score > 0.5 and class_id == 1:
            left = int(detection[3] * frame.shape[1])
            top = int(detection[4] * frame.shape[0])
            right = int(detection[5] * frame.shape[1])
            bottom = int(detection[6] * frame.shape[0])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
            cv2.putText(frame, class_labels[class_id], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

    # 顯示影像
    cv2.imshow('frame', frame)

    # 按下'q'鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()