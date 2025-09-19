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
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from pathlib import Path
import json
from piper_msgs.srv import Reset, ResetResponse

from detection.color_object_detection import ColorObjectDetector
from detection.project_bboxes_to_depth import compute_bbox_depth_means
from utils.load_paramter import load_camera_config, load_config
from inference.Sim2RealInference import Sim2RealInference
from utils.dcmm_math import rotate_B_to_A_90_xyzw


def check_ros_master():
    try:
        rosnode.rosnode_ping('rosout', max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")

def _np2py(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o

def save_obs_jsonl(filepath, obs, stamp=None):
    """将 obs 追加保存到 .jsonl 文本文件（每行一条）。"""
    rec = {"t": stamp, "obs": obs}
    line = json.dumps(rec, ensure_ascii=False, default=_np2py)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line + "\n")

class DcmmRosNode():
    """DcmmRos节点
    """
    def __init__(self) -> None:
        check_ros_master()
        rospy.init_node('dcmm_node', anonymous=True)

        self.is_info = False
        self.is_sim = True
        # 配置检测和获取深度模块
        self.config_path = "/home/kaizhen/rl_ws/sim2real_ws/src/rl_s2r/configs/config.yaml"
        self.is_visualize = True
        self.detector = ColorObjectDetector()
        self.detector.set_red_range()
        self.K_color, self.K_depth, self.R_color_to_depth = load_camera_config(self.config_path)
        self.last_object_pos = None
        self.last_object_ts = None
        self.sim_object_pos = None
        self.sim_object_vel = None
        self.sim_object_ts = None
        self.sim_hand_pose = None

        self.bridge = CvBridge()
        self.depth_buf = deque(maxlen=3)
        self.color_buf = deque(maxlen=3)

        # 配置机械臂
        self.arm_pose_buf = deque(maxlen=100)
        self.last_arm_pose = None
        self.last_arm_ts = None
        self.sim_arm_pose = None

        # 记录进入循环前的时间
        self.start_time = time.time()
        self.is_enable_flag = False

        # print(self.T_depth_to_arm_base)

        # 配置推理模块
        catch_pt_path = '/home/kaizhen/rl_ws/sim2real_ws/src/rl_s2r/models/catch_two_stage1.pth'
        env = None
        self.inference = Sim2RealInference(env)
        self.inference.restore_test(catch_pt_path)
        self.inference.set_eval()

        # 锁保护队列操作
        self.depth_lock = threading.Lock()
        self.color_lock = threading.Lock()
        self.arm_pose_lock = threading.Lock()
        self.sim_object_lock = threading.Lock()
        self.sim_hand_pose_lock = threading.Lock()
        self.sim_arm_pose_lock = threading.Lock()
        
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

        sub_sim_object_pose_th = threading.Thread(target=self.SubSimObjectPoseThread)
        sub_sim_object_pose_th.daemon = True
        sub_sim_object_pose_th.start()

        sub_sim_hand_pose_th = threading.Thread(target=self.SubSimHandPoseThread)
        sub_sim_hand_pose_th.daemon = True
        sub_sim_hand_pose_th.start()

        sub_sim_arm_pose_th = threading.Thread(target=self.SubSimArmPoseThread)
        sub_sim_arm_pose_th.daemon = True
        sub_sim_arm_pose_th.start()

        # 发布控制指令
        self.pub_pos_cmd = rospy.Publisher('/end_states', PosCmd, queue_size=1)
        self.pub_joint_state = rospy.Publisher('/joint_states', JointState, queue_size=1)
        self.pub_object_pos = rospy.Publisher('/object_pose', Point, queue_size=1)
        self.pub_sim_hand_cmd = rospy.Publisher('/sim_hand_cmd', Float32MultiArray, queue_size=1)
        self.pub_sim_pos_cmd = rospy.Publisher('/sim_end_states', PosCmd, queue_size=1)
        # service
        self.reset_srv = rospy.Service('reset_real', Reset, self.handle_reset_srv)

    def Process(self):
        """处理线程
        """
        rate = rospy.Rate(30)
        # 设置超时时间（秒）
        timeout = 5
        while not rospy.is_shutdown():
            if not self.is_enable_flag:
                self.reset()
                elapsed_time = time.time() - self.start_time
                if elapsed_time > timeout:
                    self.is_enable_flag = True
            else:
                start_time = time.time()
                obs = self.get_obs()
                # stamp = rospy.Time.now().to_sec()  # 若无 ROS，可用 time.time()
                # save_obs_jsonl("/home/kaizhen/rl_ws/sim2real_ws/obs_log_recive1.jsonl", obs, stamp)
                # self.reset()
                if obs is not None:
                    action_dict = self.inference.predict(obs)
                    # print(obs)
                    if not self.is_sim:
                        action_dict = self.limit_action(action_dict)
                    self.step(action_dict, obs)
                    # print("time:", time.time() - start_time)
                
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

    def SubSimObjectPoseThread(self):
        """仿真物体位姿订阅
        """
        rospy.Subscriber('/sim_object_pose', Odometry, self.sim_object_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubSimHandPoseThread(self):
        """仿真机械臂位姿订阅
        """
        rospy.Subscriber('/sim_hand_pose', Float32MultiArray, self.sim_hand_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubSimArmPoseThread(self):
        """仿真机械臂位姿订阅
        """
        rospy.Subscriber('/sim_arm_obs', Float32MultiArray, self.sim_arm_pose_callback, queue_size=1, tcp_nodelay=True)
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
            # print(self.arm_pose_buf[-1])
    def sim_object_pose_callback(self, msg):
        """仿真物体位姿回调
        """
        with self.sim_object_lock:
            self.sim_object_ts = msg.header.stamp.to_sec()
            self.sim_object_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.sim_object_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        # print(self.sim_object_pos)

    def sim_hand_pose_callback(self, msg):
        """仿真Hand位姿回调
        """
        with self.sim_hand_pose_lock:  
            self.sim_hand_pose = np.array(msg.data, dtype=np.float32)
        # print(self.sim_hand_pose)

    def sim_arm_pose_callback(self, msg):
        """仿真Arm位姿回调
        """
        with self.sim_arm_pose_lock:
            self.sim_arm_pose = np.array(msg.data, dtype=np.float32)
        # print(self.sim_arm_pose)

    def PublishCmdArm(self, arm_values):
        """发布Real机械臂控制指令
        该方法将接收到的动作字典中的位置和旋转值传递给 PosCmd 消息并发布。
        """
        pos_cmd = PosCmd()

        pos_cmd.x = arm_values[0]
        pos_cmd.y = arm_values[1]
        pos_cmd.z = arm_values[2]
        pos_cmd.roll = arm_values[3]
        pos_cmd.pitch = 1.18
        pos_cmd.yaw = 0.0
        pos_cmd.mode1 = 0
        pos_cmd.mode2 = 0

        self.pub_pos_cmd.publish(pos_cmd)

    def PublishSimCmdArm(self, arm_values):
        """发布Sim机械臂控制指令
        该方法将接收到的动作字典中的位置和旋转值传递给 PosCmd 消息并发布。
        """
        pos_cmd = PosCmd()

        pos_cmd.x = arm_values[0]
        pos_cmd.y = arm_values[1]
        pos_cmd.z = arm_values[2]
        pos_cmd.roll = arm_values[3]
        pos_cmd.pitch = 1.18
        pos_cmd.yaw = 0.0
        pos_cmd.mode1 = 0
        pos_cmd.mode2 = 0

        self.pub_sim_pos_cmd.publish(pos_cmd)

    def PublishCmdJointArm(self, joint_values):
        """发布机械臂关节控制指令
        """
        joint_state = JointState()

        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        joint_state.position = joint_values
        joint_state.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_state.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.pub_joint_state.publish(joint_state)

    def PublishObjectPos(self, object_pos):
        """发布物体位置
        """
        object_msg = Point()
        object_msg.x = object_pos[0]
        object_msg.y = object_pos[1]
        object_msg.z = object_pos[2]
        self.pub_object_pos.publish(object_msg)

    def PublishCmdSimHand(self, hand_values):
        """发布仿真手控制指令
        """
        hand_cmd = Float32MultiArray()
        hand_cmd.data = hand_values
        self.pub_sim_hand_cmd.publish(hand_cmd)
        # print(hand_values)

    def get_object_pos_vel(self):
        """获取物体位置和速度"""
        object_pos = (0.0, 0.0, 0.0)
        object_vel = (0.0, 0.0, 0.0)

        if self.is_sim:
            if self.sim_object_ts is not None:
                object_pos = self.sim_object_pos
                # object_vel = self.sim_object_vel
                object_ts = self.sim_object_ts
                
                if self.last_object_pos is None or self.last_object_ts is None:
                    object_vel = (0.0, 0.0, 0.0)
                else:
                    dt = object_ts - self.last_object_ts
                    if dt > 1e-6:
                        object_vel = tuple((p - lp)/dt for p, lp in zip(object_pos, self.last_object_pos))
                    else:
                        object_vel = (0.0, 0.0, 0.0)
                self.last_object_pos = object_pos
                self.last_object_ts = object_ts
                return object_pos, object_vel, object_ts
            else:
                return None, None, None
        else:
            # 获取最新的深度数据
            with self.depth_lock:
                if len(self.depth_buf) == 0:
                    if self.is_info:
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
                if self.is_info:
                    print("No color data available.")
                return None, None, None

            ts_color, color_image = color_data

            depth_map_copy = depth_map.copy() if hasattr(depth_map, 'copy') else depth_map
            color_image_copy = color_image.copy() if hasattr(color_image, 'copy') else color_image

            # 进行目标检测和深度计算
            color_detected_img, contours, bboxes = self.detector.detect(color_image_copy, min_area=300, min_aspect_ratio=0.3)
            results = compute_bbox_depth_means(bboxes, depth_map_copy, self.K_color, self.K_depth, self.R_color_to_depth, depth_filter_percentile=50)
            if len(results) == 0:
                return None, None, None
            else:
                object_pos = results[0]['center_3d']
                # 将物体坐标系转换到机械臂坐标系
                object_pos = self._trans_object_to_arm_base(object_pos)
                print("object_pos", object_pos)

                if self.last_object_pos is None or self.last_object_ts is None:
                    object_vel = (0.0, 0.0, 0.0)
                else:
                    dt = ts_depth - self.last_object_ts
                    if dt > 1e-6:
                        object_vel = tuple((p - lp)/dt for p, lp in zip(object_pos, self.last_object_pos))
                    else:
                        object_vel = (0.0, 0.0, 0.0)

                self.last_object_pos = object_pos
                self.last_object_ts = ts_depth
            
            # 可视化处理
            if self.is_visualize:
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
        # if self.is_sim:
        #     with self.sim_arm_pose_lock:
        #         if self.sim_arm_pose is not None:
        #             with self.sim_hand_pose_lock:
        #                 arm_pos = self.sim_arm_pose[:3]
        #                 quaternion = self.sim_arm_pose[3:7]
        #                 arm_vel = self.sim_arm_pose[7:]
        #             return arm_pos, quaternion, arm_vel, cur_ts
        #         else:
        #             return None, None, None, None
        # else:
            
        if len(self.arm_pose_buf) == 0:
            if self.is_info:
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
            if self.is_info:
                print("No object data available.")
            return None
        
        if self.sim_hand_pose is None:
            if self.is_info:
                print("No hand data available.")
            return None

        # 获取机械臂末端位姿
        xyz, quaternion, arm_vel, ts_arm = self.get_end_effector_pos_vel(ts_depth)
        if xyz is None or quaternion is None:
            if self.is_info:
                print("No arm pose data available.")
            return None
        # if not self.is_sim:
        # sim中的机械臂末端和真实的机械臂末端的姿态不一致，需要进行转换
        quaternion = -rotate_B_to_A_90_xyzw(quaternion)

        # 限制观测空间
        object_pos = np.clip(object_pos, -10, 10)
        object_vel = np.clip(object_vel, -4, 4)
        # object_pos = [0.3, -0.2, 1.0]
        # object_vel = [-0.1, 0.0, -0.1]
        # object_vel = np.clip(object_vel, -4, 4)
        xyz = np.clip(xyz, -10, 10)
        arm_vel = np.clip(arm_vel, -1, 1)
        # quaternion = np.clip(quaternion, -1, 1)

        object_data = {
            'pos3d': np.array([object_pos]),  # 存储物体位置
            'v_lin_3d': np.array([object_vel])  # 存储物体速度
        }
        # if self.is_sim:
        #     arm_pose = {
        #         'ee_pos3d': np.array([[xyz[0], xyz[1], xyz[2]]]),
        #         'ee_quat': np.array([[quaternion[0], quaternion[1], quaternion[2], quaternion[3]]]),
        #         'ee_v_lin_3d': np.array([[arm_vel[0], arm_vel[1], arm_vel[2]]])
        #     }
        # else:
        arm_pose = {
            'ee_pos3d': np.array([[xyz[0], xyz[1], xyz[2]]]),
            'ee_quat': np.array([[quaternion[3], quaternion[0], quaternion[1], quaternion[2]]]),
            'ee_v_lin_3d': np.array([[arm_vel[0], arm_vel[1], arm_vel[2]]])
        }

        obs = {'arm': arm_pose,
            'hand': self.sim_hand_pose[np.newaxis, :],
            'object': object_data,
        }
        return obs

    def get_action(self, obs):
        """推理模块
        """
        action_dict = self.inference.predict(obs)
        return action_dict

    def step(self, action_dict, obs):
        """单步执行
        """
        if self.is_sim:
            arm_values = action_dict["arm"].flatten()
            self.PublishSimCmdArm(arm_values)

            # TODO: 灵巧手控制
            hand_values = action_dict["hand"].flatten()
            self.PublishCmdSimHand(hand_values)

            # TODO: 发布物体坐标
            object_pos = obs["object"]["pos3d"].flatten()
            self.PublishObjectPos(object_pos)
            arm_values = action_dict["arm"].flatten()
            arm_values[0] = arm_values[0] + self.last_arm_pose[0]
            arm_values[1] = arm_values[1] + self.last_arm_pose[1]
            arm_values[2] = arm_values[2] + self.last_arm_pose[2]
            arm_values[3] = arm_values[3] + self.last_arm_pose[3]
            self.PublishCmdArm(arm_values)
        else:
            pass
    
        # TODO: 灵巧手控制


    
    def reset(self):
        """重置
        """
        # 末端位姿控制无法进行大幅的位姿控制，因此需要使用关节控制
        init_arm_joints = load_config(self.config_path, {'init_arm_joints': np.array})['init_arm_joints']
        self.PublishCmdJointArm(init_arm_joints)
        # print("Reset arm to initial Joint.")
        # TODO: 灵巧手控制

    def limit_action(self, action_dict):
        """限制动作
        --限制机械臂和灵巧手的位姿
        """
        xyz_upper_limit = load_config(self.config_path, {'xyz_upper_limit': np.array})['xyz_upper_limit'].flatten()
        xyz_lower_limit = load_config(self.config_path, {'xyz_lower_limit': np.array})['xyz_lower_limit'].flatten()

        arm_values = action_dict["arm"].flatten()

        xyz_target = arm_values[:3] + self.last_arm_pose[:3]

        # 判断超限并置 0
        for i, axis in enumerate(['x', 'y', 'z']):
            if xyz_target[i] > xyz_upper_limit[i] or xyz_target[i] < xyz_lower_limit[i]:
                arm_values[i] = 0.0  # 超范围置 0
                if self.is_info:
                    rospy.logwarn(f"[ALERT] {axis}-axis out of limit: {xyz_target[i]:.3f}, "
                            f"reset to 0 (limit range: {xyz_lower_limit[i]:.3f} ~ {xyz_upper_limit[i]:.3f})")

        action_dict["arm"] = arm_values.reshape(action_dict["arm"].shape)
        # TODO: 灵巧手控制
        return action_dict


    def _trans_object_to_arm_base(self, object_pos):
        """将物体位置转换到机械臂基坐标系
        """
        T_depth_to_arm_base = load_config(self.config_path, {'T_depth_to_arm_base': np.array})['T_depth_to_arm_base']
        object_pos = np.array(object_pos)
        object_pos = np.array([object_pos[0], object_pos[1], object_pos[2], 1.0])
        object_pos = T_depth_to_arm_base.dot(object_pos)
        return object_pos[:3]

    def handle_reset_srv(self, req):
        """重置服务
        """
        if (req.reset_request):
            # 重置初始时间戳
            self.start_time = time.time()
            # 设置重置标志位
            self.is_enable_flag = False

            response = True
            rospy.loginfo(f"Returning response: {response}")
        else:
            response = False
            rospy.loginfo(f"Returning response: {response}")

        return ResetResponse(response)

if __name__ == '__main__':
    try:
        dcmm_ros_node = DcmmRosNode()
        dcmm_ros_node.Process()
    except rospy.ROSInterruptException:
        pass
