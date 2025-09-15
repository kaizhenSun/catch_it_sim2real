#!/usr/bin/env python3

import os
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
import time
import threading
from piper_msgs.msg import PiperStatusMsg, PosCmd, PiperEulerPose
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import deque, OrderedDict
import cv2
from tf.transformations import quaternion_from_euler

from detection.color_object_detection import ColorObjectDetector
from detection.project_bboxes_to_depth import compute_bbox_depth_means
from utils.load_paramter import load_camera_config



def check_ros_master():
    try:
        rosnode.rosnode_ping('rosout', max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")

class DcmmRosNode():
    """DcmmRos节点
    """
    def __init__(self) -> None:
        check_ros_master()
        rospy.init_node('dcmm_node', anonymous=True)
        # 配置检测和获取深度模块
        self.config_path = "/home/kaizhen/rl_ws/sim2real_ws/src/rl_s2r/configs/config.yaml"
        self.visualize = False
        self.detector = ColorObjectDetector()
        self.detector.set_red_range()
        self.K_color, self.K_depth, self.R_color_to_depth = load_camera_config(self.config_path)
        self.last_object_pos = None
        self.last_object_ts = None

        self.bridge = CvBridge()
        self.depth_buf = deque(maxlen=3)
        self.color_buf = deque(maxlen=3)

        # 配置机械臂
        self.arm_pose_buf = deque(maxlen=100)
        self.last_arm_pose = None
        self.last_arm_ts = None

        # 锁保护队列操作
        self.depth_lock = threading.Lock()
        self.color_lock = threading.Lock()
        self.arm_pose_lock = threading.Lock()
        
        # 启动订阅线程
        sub_depth_th = threading.Thread(target=self.SubDepthThread)
        sub_depth_th.daemon = True
        sub_depth_th.start()

        sub_color_th = threading.Thread(target=self.SubColorThread)
        sub_color_th.daemon = True
        sub_color_th.start()

        sub_arm_end_pose_th = threading.Thread(target=self.SubArmEndPoseThread)
        sub_arm_end_pose_th.daemon = True
        sub_arm_end_pose_th.start()

    def Process(self):
        """处理线程
        """
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # object_pos, object_vel = self.get_object_pos_vel()
            # print("object_pos:", object_pos)
            # print("object_vel:", object_vel)
            obs = self.get_obs()
            # if obs is not None:
            #     print("obs:", obs)

            rate.sleep()

    def SubDepthThread(self):
        """深度图像订阅
        """
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubColorThread(self):
        """彩色图像订阅
        """
        rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubArmEndPoseThread(self):
        """机械臂末端位姿订阅
        """
        rospy.Subscriber('/end_pose_euler', PiperEulerPose, self.arm_end_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def depth_callback(self, msg):
        """深度图像回调
        """
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn(f"cv_bridge error: {e}")
            return
        
        img_u16 = np.uint16(img)
        ts = msg.header.stamp.to_sec()
        with self.depth_lock:
            self.depth_buf.append((ts, img_u16.copy()))
        
    def color_callback(self, msg):
        """彩色图像回调
        """
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logwarn(f"cv_bridge error: {e}")
            return

        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ts = msg.header.stamp.to_sec()
        with self.color_lock:
            self.color_buf.append((ts, image.copy()))

    def arm_end_pose_callback(self, msg):
        """机械臂末端位姿回调
        """
        ts = msg.header.stamp.to_sec()
        with self.arm_pose_lock:
            self.arm_pose_buf.append((ts, msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw))

    def get_object_pos_vel(self):
        """获取物体位置和速度"""
        object_pos = (0.0, 0.0, 0.0)
        object_vel = (0.0, 0.0, 0.0)
        # 获取最新的深度数据
        with self.depth_lock:
            if len(self.depth_buf) == 0:
                print("No depth data available.")
                return None, None, None
            ts_depth, depth_map = self.depth_buf[-1]
        
        # 获取时间差最小的彩色数据
        color_data = None
        min_time_diff = float('inf')
        with self.color_lock:
            for ts_color, color_img in self.color_buf:
                time_diff = abs(ts_color - ts_depth)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    color_data = (ts_color, color_img)

        if color_data is None:
            print("No color data available.")
            return None, None, None

        ts_color, color_image = color_data

        depth_map_copy = depth_map.copy() if hasattr(depth_map, 'copy') else depth_map
        color_image_copy = color_image.copy() if hasattr(color_image, 'copy') else color_image

        # 进行目标检测和深度计算
        color_detected_img, contours, bboxes = self.detector.detect(color_image_copy, min_area=300, min_aspect_ratio=0.3)
        results = compute_bbox_depth_means(bboxes, depth_map_copy, self.K_color, self.K_depth, self.R_color_to_depth, depth_filter_percentile=50)
        if len(results) > 0:
            object_pos = results[0]['center_3d']

            if self.last_object_pos is None or self.last_object_ts is None:
                object_vel = (0.0, 0.0, 0.0)
            else:
                dt = ts_depth - self.last_object_ts
                if dt > 1e-6:
                    object_vel = tuple((p - lp)/dt for p, lp in zip(object_pos, self.last_object_pos))
                else:
                    object_vel = (0.0, 0.0, 0.0)

            # 更新上一帧的位置和时间
            self.last_object_pos = object_pos
            self.last_object_ts = ts_depth
            # print("object_vel:", object_vel)
        
        # 可视化处理
        if self.visualize:
            cv2.imshow("Color Image with Detected Objects", color_detected_img)
            # if len(results) > 0:
            #     print("results:", results[0]['mean_depth'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return object_pos, object_vel, ts_depth
        else:
            return object_pos, object_vel, ts_depth
        
        return object_pos, object_vel, ts_depth

    def get_end_effector_pos_vel(self, cur_ts):
        """获取机械臂末端位置和速度"""
        if len(self.arm_pose_buf) == 0:
            print("No arm pose data available.")
            return None, None, None, None

        min_time_diff = float('inf')
        closest_pose = None

        with self.arm_pose_lock:
            for ts, x, y, z, roll, pitch, yaw in self.arm_pose_buf:
                time_diff = abs(ts - cur_ts)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_pose = (ts, x, y, z, roll, pitch, yaw)

        if closest_pose:
            ts, x, y, z, roll, pitch, yaw = closest_pose

            if self.last_arm_pose is None or self.last_arm_ts is None:
                arm_vel = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                dt = ts - self.last_arm_ts
                if dt > 1e-6:
                    arm_vel = tuple((p - lp) / dt for p, lp in zip((x, y, z, roll, pitch, yaw), self.last_arm_pose))
                else:
                    arm_vel = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            self.last_arm_pose = (x, y, z, roll, pitch, yaw)
            self.last_arm_ts = ts
            quaternion = quaternion_from_euler(roll, pitch, yaw)

            return (x, y, z), quaternion, arm_vel, ts

        return None, None, None, None
        

    def get_obs(self):
        """获取观测值
        """
        # 物体位置和速度
        object_pos, object_vel, ts_depth = self.get_object_pos_vel()
        if ts_depth is None:
            print("No object data available.")
            return None
        # 获取机械臂末端位姿
        xyz, quaternion, arm_vel, ts_arm = self.get_end_effector_pos_vel(ts_depth)
        if xyz is None or quaternion is None:
            print("No arm pose data available.")
            return None
        
        # 限制观测空间
        object_pos = np.clip(object_pos, -10, 10)
        object_vel = np.clip(object_vel, -4, 4)
        xyz = np.clip(xyz, -10, 10)
        arm_vel = np.clip(arm_vel, -1, 1)
        quaternion = np.clip(quaternion, -1, 1)

        object_data = OrderedDict([
            ('pos3d', np.array([object_pos], dtype=np.float32)),  # 存储物体位置
            ('v_lin_3d', np.array([object_vel], dtype=np.float32))  # 存储物体速度
        ])
        arm_pose = {
            'ee_pos3d': np.array([[xyz[0], xyz[1], xyz[2]]], dtype=np.float32),
            'ee_quat': np.array([[quaternion[3], quaternion[0], quaternion[1], quaternion[2]]], dtype=np.float32),
            'ee_v_lin_3d': np.array([[arm_vel[0], arm_vel[1], arm_vel[2]]], dtype=np.float32)
        }

        obs = {'arm': arm_pose,
            'hand': np.array([[2.3520093, -0.10279789, 0.9765025, 1.8407148, -0.54845285, 
                                0.01487582, 2.341205, -0.49063003, -0.21091968, -0.66344714, 
                                -1.2759252, -1.3659493]], dtype=np.float32),
            'object': object_data,
        }
        return obs


if __name__ == '__main__':
    try:
        dcmm_ros_node = DcmmRosNode()
        dcmm_ros_node.Process()
    except rospy.ROSInterruptException:
        pass
