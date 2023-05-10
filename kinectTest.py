import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config

def main():
    # 配置和啟動Kinect設備
    config = Config(
        color_resolution=pyk4a.ColorResolution.OFF,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=False,
    )
    k4a = PyK4A(config)
    try:
        k4a.start()
    except pyk4a.K4AException as e:
        print(f"Failed to start Azure Kinect device: {e}")
        return

    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
    
    min_depth = 500
    max_depth = 800

    while True:
        # 獲取深度圖像
        capture = k4a.get_capture()
        depth_image = capture.depth
        
        # 將深度圖像轉換為CV_8U
        #depth_image_8u = cv2.convertScaleAbs(depth_image, alpha=0.005)
        
         # 應用中值濾波器
        depth_image_filtered = cv2.medianBlur( depth_image , 5)

       # 過濾特定深度範圍的物體
        depth_image_filtered = np.where((depth_image_filtered > min_depth ) & (depth_image_filtered < max_depth ), depth_image_filtered, 0)

        # 將深度圖像轉換為可顯示的格式
        depth_image_filtered = cv2.convertScaleAbs(depth_image_filtered, alpha=0.05)
       
        # 應用腐蝕操作
        #kernel = np.ones((3, 3), np.uint8)
        #depth_image_eroded = cv2.erode(depth_image_filtered, kernel, iterations=1)

        # 應用膨脹操作
        #depth_image_filtered = cv2.dilate(depth_image_eroded, kernel, iterations=1)

        # 過濾特定深度範圍的物體
        #depth_image_filtered = np.where((depth_image_dilated > min_depth // 256) & (depth_image_dilated < max_depth // 256), depth_image_dilated, 0) 
       
        # 輪廓檢測
        contours, _ = cv2.findContours(depth_image_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 繪製輪廓
        #cv2.drawContours(depth_image_display, contours, -1, (0, 255, 0), 2)
        
        for contour in contours:
            # 繪製輪廓
            cv2.drawContours(depth_image_filtered, [contour], -1, (100, 255, 100), 2)

            # 計算邊界矩形並繪製
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(depth_image_filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 顯示輪廓數量
        contour_count = len(contours)
        cv2.putText(depth_image_filtered, f"Contours: {contour_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)

        # 顯示深度圖像
        cv2.imshow("Depth Image", depth_image_filtered)

        # 按下'q'鍵退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 釋放資源並關閉視窗
    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()