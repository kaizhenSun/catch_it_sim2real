import numpy as np
import cv2
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rosbag




def compute_bbox_depth_means(bboxes, depth_map, K_color, K_depth, R_color_to_depth, depth_filter_percentile=50):
    """
    计算彩色图像中边界框投影到深度图的平均深度，并返回投影信息和中心点的3D坐标

    参数:
    bboxes: 彩色图像中的边界框列表，格式为[(x, y, w, h), ...]
    depth_map: 深度图（通过cv2.IMREAD_UNCHANGED读取的原始深度图）
    K_color: 彩色相机内参矩阵 (3x3)
    K_depth: 深度相机内参矩阵 (3x3)
    R_color_to_depth: 从彩色相机到深度相机的旋转矩阵 (3x3)
    depth_filter_percentile: 用于深度图过滤的百分比（例如50表示中间50%的值）

    返回:
    results: 列表，每个元素为字典，包含:
        - mean_depth: 平均深度值
        - projected_bbox: 投影后的边界框坐标 (x, y, w, h)
        - valid_depth_points: 有效深度点的坐标和深度值列表 [(u, v, depth), ...]
        - center_3d: 中心点的3D坐标 (X, Y, Z)
    """
    
    # 确保深度图是单通道
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]  # 提取单通道数据
        
    # 获取深度图尺寸
    depth_h, depth_w = depth_map.shape
    
    # 计算从彩色相机到深度相机的投影变换矩阵
    H = K_depth @ R_color_to_depth @ np.linalg.inv(K_color)
    
    results = []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        
        # 计算彩色图像中的中心点
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 在边界框内采样多个点（不仅仅是中心点）
        sample_points = []
        
        # 计算边界框内的总点数
        total_pixels = w * h
        
        if total_pixels <= 100:
            # 如果点数小于等于100，采样所有点
            for i in range(w):
                for j in range(h):
                    sample_points.append([x + i, y + j])
        else:
            # 如果点数大于100，均匀采样100个点
            # 计算采样步长
            step_x = max(1, int(w / 10))  # 在x方向采样约10个点
            step_y = max(1, int(h / 10))  # 在y方向采样约10个点
            
            # 确保采样点数接近100
            for i in range(0, w, step_x):
                for j in range(0, h, step_y):
                    sample_points.append([x + i, y + j])
                    
                    # 如果已经采样了100个点，提前退出
                    if len(sample_points) >= 100:
                        break
                if len(sample_points) >= 100:
                    break
        
        sample_points = np.array(sample_points, dtype=np.float32)
        # print("sample_points.shape:", sample_points.shape)
        
        # 将采样点从彩色图像坐标转换到深度图像坐标
        homogeneous_pts = np.vstack([sample_points.T, np.ones(sample_points.shape[0])])
        transformed_pts = H @ homogeneous_pts
        transformed_pts = transformed_pts / transformed_pts[2, :]  # 齐次坐标归一化
        
        # 提取转换后的2D坐标
        depth_coords = transformed_pts[:2, :].T
        
        # 收集有效的深度点和坐标
        valid_depth_points = []
        valid_coords = []  # 用于计算边界框
        
        for coord in depth_coords:
            u, v = coord
            u_int = int(round(u))
            v_int = int(round(v))
            
            if 0 <= u_int < depth_w and 0 <= v_int < depth_h:
                depth_value = depth_map[v_int, u_int]   # 转换为米
                if depth_value > 0:  # 有效的深度值
                    valid_depth_points.append((u_int, v_int, float(depth_value)))
                    valid_coords.append([u, v])
        
        # 计算平均深度
        if valid_depth_points:
            # 获取所有有效深度值
            depth_values = [point[2] for point in valid_depth_points]
            
            # 根据百分比过滤来选取中间部分的值
            if len(depth_values) > 1:
                lower_percentile = np.percentile(depth_values, (100 - depth_filter_percentile) / 2)
                upper_percentile = np.percentile(depth_values, 100 - (100 - depth_filter_percentile) / 2)
                
                # 过滤掉超出中间百分比范围的深度值
                filtered_depth_values = [d for d in depth_values if lower_percentile <= d <= upper_percentile]
                
                # 计算过滤后的深度的平均值
                mean_depth = np.mean(filtered_depth_values) if filtered_depth_values else np.mean(depth_values)
            else:
                mean_depth = depth_values[0]
            
            # 计算投影后的边界框
            if valid_coords:
                valid_coords = np.array(valid_coords)
                min_u = np.min(valid_coords[:, 0])
                min_v = np.min(valid_coords[:, 1])
                max_u = np.max(valid_coords[:, 0])
                max_v = np.max(valid_coords[:, 1])
                
                projected_bbox = (
                    int(round(min_u)),
                    int(round(min_v)),
                    int(round(max_u - min_u)),
                    int(round(max_v - min_v))
                )
            else:
                projected_bbox = (0, 0, 0, 0)
        else:
            mean_depth = 0.0
            projected_bbox = (0, 0, 0, 0)
        
        mean_depth_meters = mean_depth / 1000.0  # 转换为米
        
        # 计算中心点的3D坐标
        # 将中心点从彩色图像坐标转换到深度图像坐标
        center_homogeneous = np.array([center_x, center_y, 1.0])
        center_transformed = H @ center_homogeneous
        center_transformed = center_transformed / center_transformed[2]  # 齐次坐标归一化
        
        # 使用深度相机内参和平均深度计算3D坐标
        u_center, v_center = center_transformed[:2]
        
        # 计算3D坐标 (X, Y, Z)
        # 使用深度相机内参矩阵的逆矩阵进行反投影
        K_depth_inv = np.linalg.inv(K_depth)
        point_2d_homogeneous = np.array([u_center, v_center, 1.0])
        point_3d = mean_depth_meters * (K_depth_inv @ point_2d_homogeneous)
        
        center_3d = tuple(point_3d)
        
        # print("边界框:", bbox, "平均深度:", mean_depth_meters, "投影后的边界框:", projected_bbox, "中心点3D坐标:", center_3d)
        
        # 添加到结果列表
        results.append({
            'mean_depth': mean_depth_meters,
            'projected_bbox': projected_bbox,
            'valid_depth_points': valid_depth_points,
            'center_3d': center_3d
        })
    # print("results:", results[0]['mean_depth'])
    return results


def read_image_at_index(bag_path, topic, index, bridge):
    """
    从指定rosbag文件中读取指定index的图像（可以是深度图像或彩色图像）
    
    参数：
    - bag_path: rosbag文件的路径
    - topic: 话题名称，例如 '/camera/depth/image_rect_raw' 或 '/camera/color/image_raw'
    - index: 指定的消息索引
    - bridge: cv_bridge的实例，负责将ROS图像消息转换为OpenCV图像
    
    返回：
    - image: 对应的图像（numpy数组格式）
    """
    
    # 打开rosbag文件
    with rosbag.Bag(bag_path, 'r') as bag:
        # 获取指定话题的所有消息
        messages = []
        for topic_msg, msg, t in bag.read_messages(topics=[topic]):
            messages.append(msg)
        
        # 确保索引在有效范围内
        if index < 0 or index >= len(messages):
            raise IndexError(f"Index {index} is out of range. There are {len(messages)} messages.")
        
        # 获取指定index的图像消息
        image_msg = messages[index]
        
        # 使用cv_bridge将ROS图像消息转换为OpenCV图像
        image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        
        # 如果是深度图（假设深度图采用Z16编码），则返回16位图像
        if image_msg.encoding in ['16UC1', '16SC1', 'mono16']:
            return np.uint16(image)  # 16位深度图
        # 如果是彩色图（RGB或BGR）
        elif image_msg.encoding == 'rgb8':
            # Convert RGB to BGR for OpenCV compatibility
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image  # 彩色图像（8位RGB图像转换为BGR）
        
        elif image_msg.encoding == 'bgr8':
            return image  # 彩色图像（8位BGR图像）
        
        else:
            raise ValueError(f"Unsupported image encoding: {image_msg.encoding}")

def visualize_projection(results, depth_map, K_depth, output_path=None):
    """
    可视化投影结果，包括边界框和中心点
    
    参数:
    results: compute_bbox_depth_means函数的返回结果
    depth_map: 深度图
    K_depth: 深度相机内参矩阵 (3x3)
    output_path: 输出图像路径（可选）
    """
    # 确保深度图为单通道并归一化到 0-255 的范围
    if len(depth_map.shape) == 2:  # 单通道深度图
        depth_map_normalized = np.uint8(depth_map / np.max(depth_map) * 255)  # 归一化为 0-255
    else:
        depth_map_normalized = depth_map.copy()

    # 创建彩色可视化图像（原深度图转为灰度图再转RGB）
    if len(depth_map.shape) == 2:
        vis_image = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = depth_map.copy()
    
    # 将深度图映射到颜色空间，增强深度信息的可视化效果
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # 确保两张图像有相同的通道数和数据类型
    if vis_image.shape[2] != depth_map_color.shape[2]:
        depth_map_color = cv2.cvtColor(depth_map_color, cv2.COLOR_GRAY2BGR)
    
    # 转换数据类型为 uint8
    vis_image = np.uint8(vis_image)
    depth_map_color = np.uint8(depth_map_color)

    # 将深度图映射合并到彩色图像上
    vis_image = cv2.addWeighted(vis_image, 0.6, depth_map_color, 0.4, 0)
    
    # 为每个边界框分配不同颜色
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        
        # 获取边界框的投影坐标
        x, y, w, h = result['projected_bbox']
        
        # 绘制投影后的边界框
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # 绘制有效深度点
        for point in result['valid_depth_points']:
            u, v, depth = point
            # 使用cv2.circle绘制有效深度点
            cv2.circle(vis_image, (u, v), 3, color, -1)  # 更大深度点标记
        
        # 绘制中心点
        if 'center_3d' in result:
            # 将3D坐标投影回2D图像坐标
            X, Y, Z = result['center_3d']
            # 使用深度相机内参将3D点投影到2D
            u_center = int((X * K_depth[0, 0] / Z) + K_depth[0, 2])
            v_center = int((Y * K_depth[1, 1] / Z) + K_depth[1, 2])
            
            # 绘制中心点（使用不同的颜色和更大的标记）
            cv2.circle(vis_image, (u_center, v_center), 8, (255, 255, 255), -1)  # 白色中心点
            cv2.circle(vis_image, (u_center, v_center), 6, color, -1)  # 与边界框相同颜色的内点
        
        # 添加文本信息
        text = f"Depth: {result['mean_depth']:.1f} m"
        # 在文本上添加背景色，提高可读性
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x
        text_y = y - 10
        cv2.rectangle(vis_image, (text_x, text_y - text_height), 
                      (text_x + text_width, text_y + 5), color, -1)  # 背景框
        cv2.putText(vis_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1)  # 黑色字体
        
        # 添加中心点坐标信息
        if 'center_3d' in result:
            X, Y, Z = result['center_3d']
            coord_text = f"Center: ({X:.2f}, {Y:.2f}, {Z:.2f})"
            (coord_width, coord_height), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            coord_x = x
            coord_y = y + h + 15
            cv2.rectangle(vis_image, (coord_x, coord_y - coord_height), 
                          (coord_x + coord_width, coord_y + 5), color, -1)  # 背景框
            cv2.putText(vis_image, coord_text, (coord_x, coord_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, (0, 0, 0), 1)  # 黑色字体
        
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


# 示例用法
if __name__ == "__main__":
    # 示例数据 - 实际使用时需要替换为真实数据
    color_path = '/home/kaizhen/rl_ws/data/color1.png'
    depth_pth = '/home/kaizhen/rl_ws/data/depth1.png'
    bag_path = '/home/kaizhen/rl_ws/data/2025-09-12-18-32-50.bag'
    topic1 = '/camera/depth/image_rect_raw'
    topic2 = '/camera/color/image_raw'
    index = 50
    from color_object_detection import ColorObjectDetector

    # 初始化CvBridge
    bridge = CvBridge()

    # 读取图像
    depth_map = read_image_at_index(bag_path, topic1, index, bridge)

    color_image = read_image_at_index(bag_path, topic2, index, bridge)
    # depth_map = cv2.imread(depth_pth, cv2.IMREAD_UNCHANGED)
    # depth_map = cv2.imread(depth_pth)

    detector = ColorObjectDetector()
    detector.set_red_range()
    start_time = time.time()
    color_detected_img, contours, bboxes = detector.detect(color_image, min_area=500, min_aspect_ratio=0.3)

    K_color = np.array([[611.567, 0, 323.315],
                       [0, 612.184, 242.616],
                       [0, 0, 1]])
    
    K_depth = np.array([[385.098, 0, 321.829],
                       [0, 385.098, 239.211],
                       [0, 0, 1]])
    
    R_color_to_depth = np.array([
    [0.999914, -0.0128137, -0.00282262],
    [0.0127925, 0.999891, -0.00739385],
    [0.00291705, 0.00735711, 0.999969]
    ])
    results = compute_bbox_depth_means(bboxes, depth_map, K_color, K_depth, R_color_to_depth, depth_filter_percentile=50)
    print("计算时间:", time.time() - start_time)
    # # 计算平均深度
    # depth_means = compute_bbox_depth_means(bboxes, depth_map, K_color, K_depth, R_color_to_depth)
    
    # print("边界框平均深度:", depth_means)
    # 保存结果
    vis_image = visualize_projection(results, depth_map, K_depth, "projection_result.png")
    cv2.imwrite('projection_result.png', vis_image)
    cv2.imwrite('color_detected_img.jpg', color_detected_img)