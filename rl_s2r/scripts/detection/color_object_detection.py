import cv2
import numpy as np
import time

class ColorObjectDetector:
    def __init__(self):
        """
        初始化颜色物体检测器
        """
        self.lower_color = None
        self.upper_color = None
        
    def set_red_range(self, saturation_threshold=120, value_threshold=80, hue_tolerance=5):
        """
        设置红色的颜色范围，优化阈值以减少误检测
        
        参数:
            saturation_threshold: 饱和度阈值，值越高颜色越纯（减少灰色误检）
            value_threshold: 明度阈值，值越高颜色越亮（减少黑色误检）
            hue_tolerance: 色相容差范围，控制红色范围的宽度
        """
        # 优化后的红色HSV范围
        # 第一个红色范围：0°附近
        lower_red1 = np.array([0, saturation_threshold, value_threshold], dtype=np.uint8)
        upper_red1 = np.array([hue_tolerance, 255, 255], dtype=np.uint8)
        
        # 第二个红色范围：180°附近（HSV中红色在0°和180°都有）
        lower_red2 = np.array([180 - hue_tolerance, saturation_threshold, value_threshold], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # 存储红色范围
        self.lower_color = [lower_red1, lower_red2]
        self.upper_color = [upper_red1, upper_red2]
    
    def detect(self, image, min_area=500, min_aspect_ratio=0.3, show_result=False):
        """
        检测图像中的红色物体，并筛选出连续区域的物体
        
        参数:
            image: 图像
            min_area: 最小区域面积，用于过滤小物体
            min_aspect_ratio: 最小长宽比（物体应具有一定的长宽差异）
            show_result: 是否显示结果图像
            
        返回:
            result_image: 带有检测结果的图像
            contours: 检测到的轮廓列表
        """
        if not self.lower_color or not self.upper_color:
            raise ValueError("请先设置颜色范围")
        
        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩膜（多个红色范围合成掩膜）
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in zip(self.lower_color, self.upper_color):
            mask |= cv2.inRange(hsv_image, lower, upper)
        
        # 形态学操作，去除噪声并连接连续区域
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算连接区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 开运算去噪声
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小区域和长宽比不合适的区域
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # 获取外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0

                # 筛选符合长宽比的物体（例如瓶子、杯子等）
                if aspect_ratio > min_aspect_ratio:
                    filtered_contours.append(contour)
        
        # 绘制边界框
        result_image = image.copy()
        bboxes = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_image, f"Area: {cv2.contourArea(contour):.0f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 输出矩形框的位置信息
            # print(f"矩形框位置: (x: {x}, y: {y}), 宽度: {w}, 高度: {h}, 长宽比: {float(w)/h:.2f}")
            bboxes.append((x, y, w, h))
        # 显示结果
        if show_result:
            cv2.imshow('Original Image', image)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_image, filtered_contours, bboxes

# 使用示例
if __name__ == "__main__":
    # 创建检测器实例
    detector = ColorObjectDetector()
    
    # 设置红色范围
    detector.set_red_range()
    
    # 检测图像中的红色物体，并筛选出连续区域的物体
    try:
        start_time = time.time()
        # 读取图像
        image_path = '/home/kaizhen/dvs/realsense_ws/image6.png'
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        result_img, contours, bboxes = detector.detect(image, min_area=500, min_aspect_ratio=0.3)
        print(f"检测耗时: {time.time() - start_time:.2f} 秒")
        
        # 输出检测到的物体数量
        print(f"检测到 {len(contours)} 个物体")
        print("bboxes: ", bboxes)
        
        # 保存结果
        cv2.imwrite('result_image.jpg', result_img)
        
    except Exception as e:
        print(f"错误: {e}")
