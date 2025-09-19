#!/home/kaizhen/anaconda3/envs/dcmm2/bin/python3

import os
import sys
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
import time
import threading
from piper_msgs.msg import PiperStatusMsg, PosCmd, PiperEulerPose
from piper_msgs.srv import Reset, ResetResponse
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import deque, OrderedDict
import cv2
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from pathlib import Path
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_dir, 'catch_it/gym_dcmm')
sys.path.append(relative_path)
from catch_it.gym_dcmm.envs.DcmmVecEnvArm import DcmmVecEnvArm
from catch_it.gym_dcmm.algs.ppo_dcmm.Sim2RealInference import Sim2RealInference

def check_ros_master():
    try:
        rosnode.rosnode_ping('rosout', max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")

def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def ensure_2d(x):
    """把 numpy 数组转成 (1, D)"""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[np.newaxis, :]
    return x

def obs_to_2d(obs):
    """把 obs 里的所有向量转成 (1, D)"""
    new_obs = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            new_obs[k] = obs_to_2d(v)  # 递归处理子字典
        else:
            new_obs[k] = ensure_2d(v)
    return new_obs

def ensure_1d(x):
    """把数组转成一维 (D,)；保持 dtype 不变。"""
    x = np.asarray(x)
    # 若是 (1, D) 或 (D, 1) 这类含单一维度的，squeeze 就能得到 (D,)
    # 其他情况用 ravel() 展平成一维
    if x.ndim > 1:
        x = x.squeeze()
    return x.ravel()

def dict_to_1d(d):
    """递归把字典里的数组都转成一维 (D,)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict_to_1d(v)
        else:
            out[k] = ensure_1d(v)
    return out

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

class DcmmSimRosNode():
    """DcmmSimRos节点
    """
    def __init__(self) -> None:
        check_ros_master()
        rospy.init_node('dcmm_sim_node', anonymous=True)

        self.arm_pos_cmd_lock = threading.Lock()
        self.arm_pos_cmd = deque(maxlen=1)
        self.object_pos_lock = threading.Lock()
        self.object_pos = deque(maxlen=1)
        self.hand_cmd = deque(maxlen=1)
        self.hand_cmd_lock = threading.Lock()

        # 配置推理模块
        catch_pt_path = '/home/kaizhen/rl_ws/sim2real_ws/src/rl_s2r/models/catch_two_stage1.pth'
        env = None
        self.inference = Sim2RealInference(env)
        self.inference.restore_test(catch_pt_path)
        self.inference.set_eval()

        # 环境
        self.env = DcmmVecEnvArm(task='Catching', object_name='object', render_per_step=False, 
            print_reward=False, print_info=False, 
            print_contacts=False, print_ctrl=False, 
            print_obs=False, camera_name = ["top"],
            render_mode="rgb_array", imshow_cam=False, 
            viewer = True, object_eval=False,
            env_time = 7, steps_per_policy=17)
        self.is_reset = False
        
        # TODO: 发布机械臂位姿
        # self.pub_arm_pose = rospy.Publisher('/end_pose_euler', PiperEulerPose, queue_size=1)

        # self.pub_sim_arm_obs = rospy.Publisher('/sim_arm_obs', Float32MultiArray, queue_size=1)

        # TODO: 发布物体位姿
        self.pub_object_pose = rospy.Publisher('/sim_object_pose', Odometry, queue_size=1)

        # TODO: 发布Hand的位姿
        self.pub_hand_pose = rospy.Publisher('/sim_hand_pose', Float32MultiArray, queue_size=1)

        # TODO: 订阅机械臂控制位姿
        # 订阅物体的位姿
        sub_pos_cmd_th = threading.Thread(target=self.SubPosCmdThread)
        sub_pos_cmd_th.daemon = True
        sub_pos_cmd_th.start()

        sub_hand_cmd_th = threading.Thread(target=self.SubHandCmdThread)
        sub_hand_cmd_th.daemon = True
        sub_hand_cmd_th.start()

        sub_object_pos_th = threading.Thread(target=self.SubObjectPosThread)
        sub_object_pos_th.daemon = True
        sub_object_pos_th.start()

        # service
        self.reset_srv = rospy.Service('reset_sim', Reset, self.handle_reset_srv)

    def Process(self):
        """处理线程
        """

        self.env.reset()
        rate = rospy.Rate(30)
        object_pos3d = np.array([0.8, 0, 0.2])
        spin_step = 0
        actions_dict_sim = None
        is_init = False
        while not rospy.is_shutdown():
            start_time = time.time()
            # if self.object_pos:
            #     with self.object_pos_lock:
            #         object_pos3d = np.array(self.object_pos[-1]) 
            arm_cmd = self.arm_pos_cmd[-1][0:4] if self.arm_pos_cmd else np.zeros(4)
            hand_cmd = self.hand_cmd[-1] if self.hand_cmd else np.zeros(12)
            # 如果没有没有接收到控制指令，则将控制信号置为0
            if self.arm_pos_cmd:
                self.arm_pos_cmd.popleft()
            if self.hand_cmd:
                self.hand_cmd.popleft()
            if self.object_pos:
                self.object_pos.popleft()
            
            actions_dict_sim = self.env.set_control_params(object_pos3d, arm_cmd, hand_cmd)

            # if not is_init:
            #     arm_cmd = self.arm_pos_cmd[-1][0:4] if self.arm_pos_cmd else np.zeros(4)
            #     hand_cmd = self.hand_cmd[-1] if self.hand_cmd else np.zeros(12)
            #     actions_dict_sim = env.set_control_params(object_pos3d, arm_cmd, hand_cmd)
            #     is_init = True

            obs, reward, terminated, truncated, info = self.env.step(actions_dict_sim)
            # stamp = rospy.Time.now().to_sec()  # 若无 ROS，可用 time.time()
            # save_obs_jsonl("/home/kaizhen/rl_ws/sim2real_ws/obs_log1.jsonl", obs, stamp)
            # reset_obs = obs_to_2d(obs)
            # actions_dict_sim = self.inference.predict(reset_obs)
            # actions_dict_sim = dict_to_1d(actions_dict_sim)
            # print(actions_dict_sim)

            # print("time: ", time.time() - start_time)
            # if terminated or truncated:
            #     spin_step += 1
            #     if spin_step > 20:
            #         env.reset()
            #         spin_step = 0
            if self.is_reset:
                self.env.reset()
                self.is_reset = False

            self.Publish(obs)

            rate.sleep()
    def SubPosCmdThread(self):
        """订阅机械臂控制位姿线程
        """
        rospy.Subscriber('/sim_end_states', PosCmd, self.pos_cmd_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubObjectPosThread(self):
        """订阅物体位姿线程
        """
        rospy.Subscriber('/object_pose', Point, self.object_pos_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def SubHandCmdThread(self):
        """订阅机械臂控制位姿线程
        """
        rospy.Subscriber('/sim_hand_cmd', Float32MultiArray, self.hand_cmd_callback, queue_size=1, tcp_nodelay=True)
        rospy.spin()

    def pos_cmd_callback(self, msg):
        """订阅机械臂控制位姿回调函数
        """
        with self.arm_pos_cmd_lock:
            self.arm_pos_cmd.append((msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw))
        # print(self.arm_pos_cmd)

    def object_pos_callback(self, msg):
        """订阅物体位姿回调函数
        """
        with self.object_pos_lock:
            self.object_pos.append((msg.x, msg.y, msg.z))

    def hand_cmd_callback(self, msg):
        """订阅机械臂控制位姿回调函数
        """
        with self.hand_cmd_lock:
            self.hand_cmd.append(msg.data)
            # print(self.hand_cmd)
        

    def PublishArmPose(self, obs):
        """从 obs 读取机械臂末端位姿并以欧拉角发布 (roll,pitch,yaw, 单位: 弧度)"""
        arm = obs["arm"]

        pos = np.asarray(arm["ee_pos3d"], dtype=float)
        # 四元数：obs 中是相对四元数，顺序为 wxyz（来自 MuJoCo/你的 relative_quaternion）
        quat_wxyz = normalize_quat_wxyz(arm["ee_quat"])
        # ROS 的 euler_from_quaternion 需要 xyzw，重排：
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)

        # 转欧拉角（默认 ZYX：roll=X, pitch=Y, yaw=Z）
        roll, pitch, yaw = euler_from_quaternion(quat_xyzw)

        msg = PiperEulerPose()
        msg.header.stamp = rospy.Time.now()
        msg.x, msg.y, msg.z = float(pos[0]), float(pos[1]), float(pos[2])
        msg.roll, msg.pitch, msg.yaw = float(roll), float(pitch), float(yaw)
        self.pub_arm_pose.publish(msg)

    def PublishSimObjectPose(self, obs):
        """从仿真环境读取被抛出物体的位置和速度
        """
        object = obs['object']
        pos = np.asarray(object["pos3d"], dtype=float)
        vel = np.asarray(object["v_lin_3d"], dtype=float)

        msg = Odometry()

        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])
        self.pub_object_pose.publish(msg)

    def PublishSimHandPose(self, obs):
        """从仿真环境读取Hand的位姿
        """
        msg = Float32MultiArray()
        msg.data = obs['hand'][:]
        self.pub_hand_pose.publish(msg)

    def PublishSimArmObs(self, obs):
        """从仿真环境读取机械臂的obs
        发布: ee_pos3d (3,) + ee_quat (4,) + ee_v_lin_3d (3,) → (10,)
        """
        arm = obs['arm']

        ee_pos3d = arm['ee_pos3d'].flatten()     # (3,)
        ee_quat = arm['ee_quat'].flatten()       # (4,)
        ee_v_lin_3d = arm['ee_v_lin_3d'].flatten()  # (3,)

        arm_obs = np.concatenate([ee_pos3d, ee_quat, ee_v_lin_3d])

        msg = Float32MultiArray()
        msg.data = arm_obs.tolist()
        self.pub_sim_arm_obs.publish(msg)
    
    def Publish(self, obs):
        """发布消息
        """
        # self.PublishArmPose(obs)
        self.PublishSimObjectPose(obs)
        self.PublishSimHandPose(obs)
        # self.PublishSimArmObs(obs)

    def handle_reset_srv(self, req):
        """重置服务
        """
        if (req.reset_request):
            self.is_reset = True
            response = True
            rospy.loginfo(f"Returning response: {response}")
        else:
            response = False
            rospy.loginfo(f"Returning response: {response}")

        return ResetResponse(response)

if __name__ == '__main__':
    try:
        dcmm_sim_ros_node = DcmmSimRosNode()
        dcmm_sim_ros_node.Process()
    except rospy.ROSInterruptException:
        pass
