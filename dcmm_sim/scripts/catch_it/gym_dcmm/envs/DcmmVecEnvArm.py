"""
ABOUT: this file constains the RL environment for the Arm task
"""
import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
# print(sys.path)
import argparse
import math
print(os.getcwd())
import configs.env.DcmmCfg as DcmmCfg
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
# Changed the class of the MJ_DCMM for test arm
from gym_dcmm.agents.MujocoDcmmArm import MJ_DCMM
from gym_dcmm.utils.ik_pkg.ik_base import IKBase
import copy
from termcolor import colored
from decorators import *
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from utils.util import *
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from collections import deque
import matplotlib.pyplot as plt
from termcolor import colored
import time
from tf.transformations import quaternion_from_euler, euler_from_quaternion

np.set_printoptions(precision=8)

paused = True

trigger_delta = False
trigger_delta_hand = False
delta_x = 0
delta_y = 0 
delta_z = 0

def env_key_callback(keycode):
    print(f"Keycode pressed: {keycode}")  # 更清晰的调试信息
    global trigger_delta, trigger_delta_hand, delta_x, delta_y, delta_z, delta_xyz_hand
    if keycode == 324:
        trigger_delta = True
        delta_x = 0.05
        delta_y = 0
        delta_z = 0
        print(f"delta_x set to: {delta_x}")
    if keycode == 321:
        trigger_delta = True
        delta_x = -0.05
        delta_y = 0
        delta_z = 0
        print(f"delta_x set to: {delta_x}")
    if keycode == 325:
        trigger_delta = True
        delta_x = 0
        delta_y = 0.05
        delta_z = 0
        print(f"delta_y set to: {delta_y}")
    if keycode == 322:
        trigger_delta = True
        delta_x = 0
        delta_y = -0.05
        delta_z = 0
        print(f"delta_y set to: {delta_y}")
    if keycode == 326:
        trigger_delta = True
        delta_x = 0
        delta_y = 0
        delta_z = 0.05
        print(f"delta_z set to: {delta_z}")
    if keycode == 323:
        trigger_delta = True
        delta_x = 0
        delta_y = 0
        delta_z = -0.05
        print(f"delta_z set to: {delta_z}")
    if keycode == 265: # AKA 7 (on the numpad)
        trigger_delta_hand = True
        delta_xyz_hand = 0.2
    if keycode == 264: # AKA 9 (on the numpad)
        trigger_delta_hand = True
        delta_xyz_hand = -0.2

class DcmmVecEnvArm(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}
    """
    Args:
        render_mode: str
            The mode of rendering, including "rgb_array", "depth_array".
        render_per_step: bool
            Whether to render the mujoco model per simulation step.
        viewer: bool
            Whether to show the mujoco viewer.
        imshow_cam: bool
            Whether to show the camera image.
        object_eval: bool
            Use the evaluation object.
        camera_name: str
            The name of the camera.
        object_name: str
            The name of the object.
        env_time: float
            The maximum time of the environment.
        steps_per_policy: int
            The number of steps per action.
        img_size: tuple
            The size of the image.
    """
    def __init__(
        self,
        task="tracking",
        render_mode="depth_array",
        render_per_step=False,
        viewer=False,
        imshow_cam=False,
        object_eval=False,
        camera_name=["top", "wrist"],
        object_name="object",
        env_time=2.5,
        steps_per_policy=5,
        img_size=(480, 640),
        device='cuda:0',
        print_obs=False,
        print_reward=False,
        print_ctrl=False,
        print_info=False,
        print_contacts=False,
    ):
        if task not in ["Tracking", "Catching"]:
            raise ValueError("Invalid task: {}".format(task))
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.img_size = img_size
        self.device = device
        self.steps_per_policy = steps_per_policy
        self.render_per_step = render_per_step
        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts
        # Initialize the environment
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name, object_eval=object_eval)
        # self.Dcmm.show_model_info()
        self.fps = 1 / (self.steps_per_policy * self.Dcmm.model.opt.timestep)

        # Randomize the Object Info
        self.random_mass = 0.25
        self.object_static_time = 2
        self.object_throw = False
        self.object_train = True
        if object_eval: self.set_object_eval()
        self.Dcmm.model_xml_string = self._reset_object()
        self.Dcmm.model = mujoco.MjModel.from_xml_string(self.Dcmm.model_xml_string)
        self.Dcmm.data = mujoco.MjData(self.Dcmm.model)

        # Get the geom id of the hand, the floor and the object
        self.hand_start_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'mcp_joint') - 1
        # print("self.hand_start_id: ", self.hand_start_id)
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)

        # Set the camera configuration
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(
            self.Dcmm.model, self.Dcmm.data
        )
        if self.Dcmm.open_viewer:
            if self.Dcmm.viewer:
                print("Close the previous viewer")
                self.Dcmm.viewer.close()
            self.Dcmm.viewer = mujoco.viewer.launch_passive(self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)
            v = self.Dcmm.viewer
            # 世界坐标系
            v.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            # 关节坐标系
            # v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            # 相机
            v.cam.lookat[0:2] = [0, 1]
            v.cam.distance = 5.0
            v.cam.azimuth = 180
            # self.viewer.cam.elevation = -1.57
        else: self.Dcmm.viewer = None

        # Observations are dictionaries with the agent's and the object's state. (dim = 44)
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 6
        # Total dim = 34; arm = 16; hand = 12; object = 6
        self.observation_space = spaces.Dict(
            {   "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(0, 6)]),
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(0, 6)]),
                                            dtype=np.float32),
                }),
                "hand": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices]),
                    high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices]),
                    dtype=np.float32),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                }),
            }
        )
        # Define the limit for the arm action
        arm_low = -0.025*np.ones(4)
        arm_high = 0.025*np.ones(4)
        # Define the limit for the hand action
        hand_low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices])
        hand_high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices])

        # Get initial ee_pos3d
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        self.initial_obj_pos3d = self._get_relative_object_pos3d()
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]

        # Actions (dim = 16)
        # arm = 4 (x,y,z,roll); hand = 12
        self.action_space = spaces.Dict(
            {
                "arm": spaces.Box(arm_low, arm_high, shape=(4,), dtype=np.float32),
                "hand": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),
            }
        )
        self.action_buffer = {
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),
        }
        # Combine the limits of the action space
        self.actions_low = np.concatenate([arm_low, hand_low])
        self.actions_high = np.concatenate([arm_high, hand_high])

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = get_total_dimension(self.action_space)
        self.obs_t_dim = self.obs_dim - 12 - 6 # dim = 16, hand = 12, 6 for the arm joint positions
        self.act_t_dim = self.act_dim -12   # dim = 4, hand = 12 
        self.obs_c_dim = self.obs_dim - 6  # dim = 28, 6 for the arm joint positions
        self.act_c_dim = self.act_dim # dim = 16,

        # Init env params
        self.arm_limit = True
        self.terminated = False
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time
        self.reward_touch = 0
        self.reward_stability = 0
        self.env_time = env_time
        self.stage_list = ["tracking", "grasping"]
        # Default stage is "tracking"
        self.stage = self.stage_list[0]
        self.steps = 0

        self.prev_ctrl = np.zeros(18)
        self.init_ctrl = True
        self.vel_init = False
        self.vel_history = deque(maxlen=4)

        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "env_time": self.Dcmm.data.time - self.start_time,
            "imgs": {}
        }
        self.contacts = {
            # Get contact point from the mujoco model
            "object_contacts": np.array([]),
            "hand_contacts": np.array([]),
        }

        self.object_q = np.array([1, 0, 0, 0])
        self.object_pos3d = np.array([0, 0, 1.5])
        self.object_vel6d = np.array([0., 0., 1.25, 0.0, 0.0, 0.0])
        self.step_touch = False

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))

        # Random PID Params
        self.k_arm = np.ones(6)
        self.k_hand = np.ones(1)
        # Random Obs & Act Params
        self.k_obs_arm = DcmmCfg.k_obs_arm
        self.k_obs_hand = DcmmCfg.k_obs_hand
        self.k_obs_object = DcmmCfg.k_obs_object
        self.k_act = DcmmCfg.k_act

        # 调试PID
        self.PID_debug = False
        self.time_his = []
        self.joint_his = []
        self.joint_target_his = []
        self.joint_ctrl_his = []
        

    def set_object_eval(self):
        self.object_train = False

    def update_render_state(self, render_per_step):
        self.render_per_step = render_per_step

    def update_stage(self, stage):
        if stage in self.stage_list:
            self.stage = stage
        else:
            raise ValueError("Invalid stage: {}".format(stage))

    def _get_contacts(self):
        # Contact information of the hand
        geom_ids = self.Dcmm.data.contact.geom
        geom1_ids = self.Dcmm.data.contact.geom1
        geom2_ids = self.Dcmm.data.contact.geom2
        ## get the contact points of the hand
        geom1_hand = np.where((geom1_ids < self.object_id) & (geom1_ids >= self.hand_start_id))[0]
        geom2_hand = np.where((geom2_ids < self.object_id) & (geom2_ids >= self.hand_start_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_hand.size != 0:
            contacts_geom1 = geom_ids[geom1_hand][:,1]
        if geom2_hand.size != 0:
            contacts_geom2 = geom_ids[geom2_hand][:,0]
        # 这里面也包含了手指之间的接触，手与物体的接触，以及手与机械臂的接触
        hand_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the object
        geom1_object = np.where((geom1_ids == self.object_id))[0]
        geom2_object = np.where((geom2_ids == self.object_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom_ids[geom1_object][:,1]
        if geom2_object.size != 0:
            contacts_geom2 = geom_ids[geom2_object][:,0]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the floor
        # Get the contact points of the floor (self.floor_id)
        geom1_floor = np.where(geom1_ids == self.floor_id)[0]
        geom2_floor = np.where(geom2_ids == self.floor_id)[0]
        contacts_geom1_floor = np.array([]); contacts_geom2_floor = np.array([])

        if geom1_floor.size != 0:
            contacts_geom1_floor = geom_ids[geom1_floor][:,1]
        if geom2_floor.size != 0:
            contacts_geom2_floor = geom_ids[geom2_floor][:,0]

        # Concatenate floor contact points
        floor_contacts = np.concatenate((contacts_geom1_floor, contacts_geom2_floor))

        # Remove hand contacts that are also in contact with the floor
        hand_contacts_with_arm_floor_contacts = hand_contacts[hand_contacts < self.hand_start_id]

        # Now, floor contacts should only include those with the hand
        # floor_contacts_with_hand = floor_contacts[(floor_contacts < self.object_id) & (floor_contacts >= self.hand_start_id)]
        # self.print_contacts = True
        if self.print_contacts:
            print("object_contacts: ", object_contacts)
            print("floor_contacts: ", floor_contacts)
            print("hand_contacts: ", hand_contacts)
            print("hand_contacts_with_arm_floor_contacts: ", hand_contacts_with_arm_floor_contacts)
        return {
            # Get contact point from the mujoco model
            "object_contacts": object_contacts,
            "hand_contacts": hand_contacts,
            "floor_contacts": floor_contacts,
            "hand_contacts_with_arm_floor_contacts": hand_contacts_with_arm_floor_contacts,
        }
    
    def _get_relative_ee_pos3d(self):
        """
        返回在 arm_base 坐标系下的末端(link6)位置 [x, y, z]。
        原理：p_ee_base = R_base^T * (p_ee_world - p_base_world)
        """
        p_base = self.Dcmm.data.body("arm_base").xpos.copy()  # 世界系下 arm_base 的位置 (3,)
        p_ee   = self.Dcmm.data.body("link6").xpos.copy()     # 世界系下 link6 的位置 (3,)

        # xmat 是 3x3 旋转矩阵（按行摊平存储），这里 reshape 回矩阵
        R_base = self.Dcmm.data.body("arm_base").xmat.copy().reshape(3, 3)  # 世界->底座 的旋转用 R^T
        dp_world = p_ee - p_base
        dp_base  = R_base.T @ dp_world
        return dp_base

    def _get_relative_ee_quat(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        quat = relative_quaternion(self.Dcmm.data.body("arm_base").xquat, self.Dcmm.data.body("link6").xquat)
        
        # print("self.Dcmm.data.body(arm_base).xquat: ", self.Dcmm.data.body("arm_base").xquat)
        # print("self.Dcmm.data.body(link6).xquat: ", self.Dcmm.data.body("link6").xquat)
        # print("quat: ", quat)
        return np.array(quat)

    def _get_relative_object_pos3d(self):
        """
        返回在 arm_base 坐标系下的目标物体位置 [x, y, z]。
        原理：p_obj_base = R_base^T * (p_obj_world - p_base_world)
        """
        p_base = self.Dcmm.data.body("arm_base").xpos.copy()                # 世界系下 arm_base 位置 (3,)
        p_obj  = self.Dcmm.data.body(self.Dcmm.object_name).xpos.copy()     # 世界系下 物体 位置 (3,)

        # xmat 为 3x3 旋转矩阵（按行摊平存储），表示从 arm_base 坐标系到世界系的旋转
        R_base = self.Dcmm.data.body("arm_base").xmat.copy().reshape(3, 3)

        dp_world = p_obj - p_base
        dp_base  = R_base.T @ dp_world   # 世界 -> arm_base
        return dp_base

    def _get_obs(self):
        ee_pos3d = self._get_relative_ee_pos3d()
        obj_pos3d = self._get_relative_object_pos3d()
        if self.init_pos:
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.init_pos = False
        # Add Obs Noise (Additive self.arm)
        obs = {
            "arm": {
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),
                # "ee_quat": self._get_relative_ee_quat() + np.random.normal(0, self.k_obs_arm, 4),
                "ee_quat": self._get_relative_ee_quat(),
                'ee_v_lin_3d': (ee_pos3d - self.prev_ee_pos3d)*self.fps + np.random.normal(0, self.k_obs_arm, 3),
                "joint_pos": np.array(self.Dcmm.data.qpos[0:6]) + np.random.normal(0, self.k_obs_arm, 6),
            },
            "hand": self._get_hand_obs() + np.random.normal(0, self.k_obs_hand, 12),
            "object": {
                "pos3d": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),
                "v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object, 3),
            },
        }
        self.prev_ee_pos3d = ee_pos3d
        self.prev_obj_pos3d = obj_pos3d
        if self.print_obs:
            print("##### print obs: \n", obs)
        return obs
    
    def _get_hand_obs(self):
        hand_obs = np.zeros(12)
        # Thumb
        hand_obs[9] = self.Dcmm.data.qpos[6+13]
        hand_obs[10] = self.Dcmm.data.qpos[6+14]
        hand_obs[11] = self.Dcmm.data.qpos[6+15]
        # Other Three Fingers
        hand_obs[0] = self.Dcmm.data.qpos[6]
        hand_obs[1:3] = self.Dcmm.data.qpos[(6+2):(6+4)]
        hand_obs[3] = self.Dcmm.data.qpos[6+4]
        hand_obs[4:6] = self.Dcmm.data.qpos[(6+6):(6+8)]
        hand_obs[6] = self.Dcmm.data.qpos[6+8]
        hand_obs[7:9] = self.Dcmm.data.qpos[(6+10):(6+12)]
        return hand_obs
    def _get_info(self):
        # Time of the Mujoco environment
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        if self.print_info: 
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
            print("base_distance: ", base_distance)
        return {
            # Get contact point from the mujoco model
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
        }
    
    
    def update_target_ctrl(self):
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))
        self.action_buffer["hand"].append(copy.deepcopy(self.Dcmm.target_hand_qpos[:]))

    def _get_ctrl(self):
        # Map the action to the control 
        # print("##### get ctrl", self.action_buffer["arm"][0])
        mv_arm = self.Dcmm.arm_pid.update(self.action_buffer["arm"][0], self.Dcmm.data.qpos[0:6], self.Dcmm.data.time) # 6
        mv_hand = self.Dcmm.hand_pid.update(self.action_buffer["hand"][0], self.Dcmm.data.qpos[6:22], self.Dcmm.data.time) # 16
        ctrl = np.concatenate([mv_arm, mv_hand], axis=0)
        # Add Action Noise (Scale with self.k_act)
        ctrl *= np.random.normal(1, self.k_act, 22)
        if self.print_ctrl:
            print("##### ctrl:")
            print("mv_arm: {}\n, \nmv_hand: {}\n".format(mv_arm, mv_hand))
        return ctrl
    
    def _reset_object(self):
        # Parse the XML string
        root = ET.fromstring(self.Dcmm.model_xml_string)

        # Find the <body> element with name="object"
        object_body = root.find(".//body[@name='object']")
        if object_body is not None:
            inertial = object_body.find("inertial")
            if inertial is not None:
                # Generate a random mass within the specified range
                self.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[0])
                # Update the mass attribute
                inertial.set("mass", str(self.random_mass))
            joint = object_body.find("joint")
            if joint is not None:
                # Generate a random damping within the specified range
                random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
                # Update the damping attribute
                joint.set("damping", str(random_damping))
            # Find the <geom> element
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:
                # Modify the type and size attributes
                object_id = np.random.choice([0, 1, 2, 3, 4])
                if self.object_train:
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)  # Replace "box" with the desired type
                    object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                else:
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')
        
        return xml_str
    
    def random_object_pose(self):
        # Random Position
        x = 0.8 + 0.2*np.random.rand()
        y = 0.3 * np.random.rand() - 0.15
        # Low or High Starting Position
        low_factor = False if np.random.rand() < 0.5 else True
        # low_factor = True
        if low_factor: height = 0.4 + 0.3 * np.random.rand()# (0.7, 1.0)
        else: height = 0.5 + 0.6 * np.random.rand() # (1.0, 1.6)
        # Random Velocity
        r_vel = 1 + 0.5 * np.random.rand() # (1, 2)
        alpha_vel = math.pi * (np.random.rand()*1/6 + 5/12) # alpha_vel = (5/12 * pi, 7/12 * pi)
        v_lin_x = - r_vel * math.sin(alpha_vel) # (-0.0, -0.5)
        v_lin_y = r_vel * math.cos(alpha_vel) # (-0.5, -1.0)
        v_lin_z = 2 * np.random.rand() + 1.0 # (2.0, 2.5)
        self.object_pos3d = np.array([x, y, height])
        self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
        # self.object_vel6d = np.array([0, 0, 0, 0.0, 0.0, 0.0])
        # Random Static Time
        self.object_static_time = np.random.uniform(DcmmCfg.object_static[0], DcmmCfg.object_static[1])
        # Random Quaternion
        r_obj_quat = R.from_euler('xyz', [0, np.random.rand()*1*math.pi, 0], degrees=False)
        self.object_q = r_obj_quat.as_quat()

    def random_PID(self):
        # Random the PID Controller Params in DCMM
        self.k_arm = np.random.uniform(0, 1, size=6)
        self.k_hand = np.random.uniform(0, 1, size=1)

        # Reset the PID Controller
        self.Dcmm.arm_pid.reset(self.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])
        self.Dcmm.hand_pid.reset(self.k_hand[0]*(DcmmCfg.k_hand[1]-DcmmCfg.k_hand[0])+DcmmCfg.k_hand[0])

    def random_delay(self):
        # Random the Delay Buffer Params in DCMM
        self.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        self.action_buffer["hand"].set_maxlen(np.random.choice(DcmmCfg.act_delay['hand']))
        # Clear Buffer
        self.action_buffer["arm"].clear()
        self.action_buffer["hand"].clear()

    def _reset_simulation(self):
        # Reset the data in Mujoco Simulation
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)
        self.Dcmm.data.qpos[0:6] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[6:22] = DcmmCfg.hand_joints[:]
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.body("object").xpos[0:3] = np.array([2, 2, 1])

        # Random 3D position TODO: Adjust to the fov
        self.random_object_pose()
        self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    velocity=np.zeros(6))
        # Random Gravity
        self.Dcmm.model.opt.gravity[2] = -9.81 + 0.5*np.random.uniform(-1, 1)
        # Random PID
        self.random_PID()
        # Random Delay
        self.random_delay()
        # Forward Kinematics
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)

    def reset(self):
        # Reset the basic simulation
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.object_throw = False
        self.steps = 0
        # Reset the time
        self.start_time = self.Dcmm.data.time
        self.catch_time = self.Dcmm.data.time - self.start_time

        ## Reset the target joint positions of the arm
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        ## Reset the target joint positions of the hand
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]

        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0

        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("link6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "evn_time": self.Dcmm.data.time - self.start_time,
        }
        # Get the observation and info
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d = self._get_relative_object_pos3d()
        observation = self._get_obs()
        info = self._get_info()
        # Rendering
        imgs = self.render()
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_hand, ctrl_delay))

        return observation, info

    def norm_ctrl(self, ctrl, components):
        '''
        Convert the ctrl (dict type) to the numpy array and return its norm value
        Input: ctrl, dict
        Return: norm, float
        '''
        ctrl_array = np.concatenate([ctrl[component]*DcmmCfg.reward_weights['r_ctrl'][component] for component in components])
        return np.linalg.norm(ctrl_array)

    def compute_reward(self, obs, info, ctrl):
        '''
        Rewards:
        - Object Position Reward
        - Object Orientation Reward
        - Object Touch Success Reward
        - Object Catch Stability Reward
        - Collision Penalty
        - Constraint Penalty
        '''
        
        rewards = 0.0
        reward_base_pos = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"]
        reward_ee_pos = (self.info["ee_distance"] - info["ee_distance"]) * DcmmCfg.reward_weights["r_ee_pos"]
        reward_ee_precision = math.exp(-50*info["ee_distance"]**2) * DcmmCfg.reward_weights["r_precision"]

        # TODO:计算灵巧手和地面碰撞的惩罚

        reward_collision = 0
        if self.contacts['hand_contacts_with_arm_floor_contacts'].size != 0:
            reward_collision = DcmmCfg.reward_weights["r_collision"]

        ## Constraint Penalty
        # Compute the Penalty when the arm joint position is out of the joint limits
        reward_constraint = 0 if self.arm_limit else -1
        reward_constraint *= DcmmCfg.reward_weights["r_constraint"]

                ## Object Touch Success Reward
        # Compute the reward when the object is caught successfully by the hand
        if self.step_touch:
            # print("TRACK SUCCESS!!!!!")
            if not self.reward_touch:
                self.catch_time = self.Dcmm.data.time - self.start_time
            self.reward_touch = DcmmCfg.reward_weights["r_touch"][self.task]
        else:
            self.reward_touch = 0

        if self.task == "Catching":
            reward_orient = 0
            ## Calculate the total reward in different stages
            if self.stage == "tracking":
                ## Ctrl Penalty
                # Compute the norm of hand joint movement through the current actions in the tracking stage
                reward_ctrl = - self.norm_ctrl(ctrl, {"hand"})
                ## Object Orientation Reward
                # Compute the dot product of the velocity vector of the object and the z axis of the end_effector
                rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
                hand_z_axis = np.array([0, 0, 1])
                reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
                ## Add up the rewards
                rewards = reward_base_pos + reward_ee_pos + reward_orient + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
                if self.print_reward:
                    if reward_constraint < 0:
                        print("ctrl: ", ctrl)
                    print("### print reward")
                    print("reward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, reward_ctrl: {:.3f}, \n".format(
                        reward_ee_pos, reward_ee_precision, reward_orient, reward_ctrl
                    ) + "reward_collision: {:.3f}, reward_constraint: {:.3f}, reward_touch: {:.3f}".format(
                        reward_collision, reward_constraint, self.reward_touch
                    ))
                    print("total reward: {:.3f}\n".format(rewards))
            else:
                ## Ctrl Penalty
                # Compute the norm of base and arm movement through the current actions in the grasping stage
                reward_ctrl = - self.norm_ctrl(ctrl, {"arm"})
                ## Set the Orientation Reward to maximum (1)
                reward_orient = 1
                ## Object Touch Stability Reward
                # Compute the reward when the object is caught stably in the hand
                if self.reward_touch:
                    self.reward_stability = (info["env_time"] - self.catch_time) * DcmmCfg.reward_weights["r_stability"]
                else:
                    self.reward_stability = 0.0
                ## Add up the rewards
                rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ctrl + reward_collision + reward_constraint \
                        + self.reward_touch + self.reward_stability
                if self.print_reward:
                    print("##### print reward")
                    print("reward_touch: {}, \nreward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, \n".format(
                        self.reward_touch, reward_ee_pos, reward_ee_precision, reward_orient
                    ) + "reward_stability: {:.3f}, reward_collision: {:.3f}, \nreward_ctrl: {:.3f}, reward_constraint: {:.3f}".format(
                        self.reward_stability, reward_collision, reward_ctrl, reward_constraint
                    ))
                    print("total reward: {:.3f}\n".format(rewards))
        elif self.task == 'Tracking':
            ## Ctrl Penalty
            # Compute the norm of base and arm movement through the current actions in the grasping stage
            reward_ctrl = - self.norm_ctrl(ctrl, {"arm"})
            ## Object Orientation Reward
            # Compute the dot product of the velocity vector of the object and the z axis of the end_effector
            rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
            local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
            hand_z_axis = np.array([0, 0, 1])
            reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
            ## Add up the rewards
            rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ctrl + reward_collision + reward_constraint + self.reward_touch
            if self.print_reward:
                if reward_constraint < 0:
                    print("ctrl: ", ctrl)
                print("### print reward")
                print("reward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, reward_ctrl: {:.3f}, \n".format(
                    reward_ee_pos, reward_ee_precision, reward_orient, reward_ctrl
                ) + "reward_collision: {:.3f}, reward_constraint: {:.3f}, reward_touch: {:.3f}".format(
                    reward_collision, reward_constraint, self.reward_touch
                ))
                print("total reward: {:.3f}\n".format(rewards))
        else:
            raise ValueError("Invalid task: {}".format(self.task))

        return rewards

    def _step_mujoco_simulation(self, action_dict):
        action_arm = np.concatenate((action_dict["arm"], np.zeros(2)))
        result_QP, _ = self.Dcmm.move_ee_pose(action_arm)
        if result_QP[1]:
            self.arm_limit = True
            self.Dcmm.target_arm_qpos[:] = result_QP[0]
        else:
            # print("IK Failed!!!")
            self.arm_limit = False
        self.Dcmm.action_hand2qpos(action_dict["hand"])
        
        if self.PID_debug:
            self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
            if self.Dcmm.data.time - self.start_time > 5:

                id = 1
                f = 1
                deg = 0.7*math.sin(1/f*math.pi*self.Dcmm.data.time)
                deg = 0.2
                self.Dcmm.target_arm_qpos[id] = DcmmCfg.arm_joints[id] + deg
        

        # Add Target Action to the Buffer
        self.update_target_ctrl()
        # Reset the Criteria for Successfully Touch
        self.step_touch = False


        for i in range(self.steps_per_policy):
            # Update the control command according to the latest policy output                
            self.Dcmm.data.ctrl[:-1] = self._get_ctrl()
            # if self.PID_debug:
            #     self.Dcmm.data.ctrl[5:-1] = np.zeros(1)

            
            if self.render_per_step:
                # Rendering
                img = self.render()
            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            if self.Dcmm.data.time - self.start_time < self.object_static_time:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=np.zeros(6))
                self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
            elif not self.object_throw:
                self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                            velocity=self.object_vel6d[:])
                self.Dcmm.data.ctrl[-1] = 0.0
                self.object_throw = True

            # self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
            #                 velocity=np.zeros(6))
            # self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
            # 测试物体随机状态
            # if self.Dcmm.data.time - self.start_time > 5:
            #     self.object_throw = False
            #     self.random_object_pose()
            #     self.start_time = self.Dcmm.data.time

            mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)
            mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)
            # Update the contact information
            self.contacts = self._get_contacts()
            # Whether the base collides
            # TODO：需要考虑灵巧手和地面的接触，这种事情不允许在现实中发生
            # 除上述外还需要考虑灵巧手与除了物体和自身意外的接触，这也是不被允许的
            if self.contacts['hand_contacts_with_arm_floor_contacts'].size != 0:
                # print("Occurred collision with the floor")
                self.terminated = True
            mask_coll = self.contacts['object_contacts'] < self.hand_start_id
            mask_finger = self.contacts['object_contacts'] > self.hand_start_id
            mask_hand = self.contacts['object_contacts'] >= self.hand_start_id
            mask_palm = self.contacts['object_contacts'] == self.hand_start_id
            # Whether the object is caught
            if self.step_touch == False:
                if self.task == "Catching" and np.any(mask_hand):
                    self.step_touch = True
                elif self.task == "Tracking" and np.any(mask_palm):
                    self.step_touch = True
            # Whether the object falls
            if not self.terminated:
                if self.task == "Catching":
                    self.terminated = np.any(mask_coll)
                elif self.task == "Tracking":
                    self.terminated = np.any(mask_coll) or np.any(mask_finger)

            if self.PID_debug:                
                if i == 0:
                    self.joint_ctrl_his.append(self.Dcmm.data.ctrl[:6].copy())
                    self.time_his.append(self.Dcmm.data.time)
                    self.joint_his.append(self.Dcmm.data.qpos[:6].copy())
                    self.joint_target_his.append(self.Dcmm.target_arm_qpos.copy())
                    # print("self.Dcmm.target_arm_qpos[0]: ", self.Dcmm.target_arm_qpos[0])
                # If the object falls, terminate the episode in advance
                if self.Dcmm.data.time - self.start_time > 10:
                    self.terminated = True
            if self.terminated:
                break

    def step(self, action):
        self.steps += 1
        # start_time = time.time()
        self._step_mujoco_simulation(action)
        
        # Get the obs and info
        obs = self._get_obs()
        info = self._get_info()
        if self.task == 'Catching':
            if info['ee_distance'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                self.stage = "grasping"
            elif info['ee_distance'] >= DcmmCfg.distance_thresh and self.stage == "grasping":
                self.terminated = True

        # Design the reward function
        reward = self.compute_reward(obs, info, action)
        # print("reward: ", reward)
        self.info["ee_distance"] = info["ee_distance"]
        self.info["base_distance"] = info["base_distance"]

        # Rendering
        imgs = self.render()
        # Update the imgs
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['arm']),
                        len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_hand, ctrl_delay))
        # The episode is truncated if the env_time is larger than the predefined time
        truncated = False
        terminated = self.terminated
        # The episode is truncated if the env_time is larger than the predefined time
        if self.task == "Catching":
            if info["env_time"] > self.env_time:
                # print("Catching Success!!!!!!")
                truncated = True
            else: truncated = False
        elif self.task == "Tracking":
            if self.step_touch:
                # print("Tracking Success!!!!!!")
                truncated = True
            else: truncated = False
        terminated = self.terminated
        # done = terminated or truncated
        # print("step time: ", time.time() - start_time)

        return obs, reward, terminated, truncated, info

    def set_control_params(self, object_pos3d, arm_pos, hand_pos):
        # Set the target position of the object
        # self.object_pos3d = object_pos3d
        # Set the target position of the arm
        action = np.zeros(16)
        action[:4] = arm_pos
        action[4:] = hand_pos

        arm_tensor = action[0:4]
        hand_tensor = action[4:16]

        actions_dict = {
            'arm': arm_tensor,
            'hand': hand_tensor,
        }
        return actions_dict
        

    def preprocess_depth_with_mask(self, rgb_img, depth_img, 
                                   depth_threshold=3.0, 
                                   num_white_points_range=(5, 15),
                                   point_size_range=(1, 5)):
        # Define RGB Filter
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        # Apply combined mask to depth image
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        # Calculate mean depth within combined mask
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        # Generate random number of white points
        num_white_points = np.random.randint(num_white_points_range[0], num_white_points_range[1])
        # Generate random coordinates for white points
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        # Generate random sizes for white points in the specified range
        random_sizes = np.random.randint(point_size_range[0], point_size_range[1], size=num_white_points)
        # Create masks for all white points at once
        y, x = np.ogrid[:masked_depth_img.shape[0], :masked_depth_img.shape[1]]
        point_masks = ((x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2) <= random_sizes ** 2
        # Update masked depth image with the white points
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        imgs_depth = np.zeros((0, self.img_size[0], self.img_size[1]))
        for camera_name in self.camera_name:
            if self.render_mode == "human":
                self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                return imgs
            elif self.render_mode != "depth_rgb_array":
                img = self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                if self.imshow_cam and self.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # Converts the depth array valued from 0-1 to real meters
                elif self.render_mode == "depth_array":
                    img = self.Dcmm.depth_2_meters(img)
                    if self.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                        cv.imshow(camera_name+"_depth", depth_norm)
                        cv.waitKey(1)
                    img = np.expand_dims(img, axis=0)
            else:
                img_rgb = self.mujoco_renderer.render(
                    "rgb_array", camera_name = camera_name
                )
                img_depth = self.mujoco_renderer.render(
                    "depth_array", camera_name = camera_name
                )   
                # Converts the depth array valued from 0-1 to real meters
                img_depth = self.Dcmm.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.imshow_cam:
                    cv.imshow(camera_name+"_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
                    cv.imshow(camera_name+"_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.img_size[1], self.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            # Sync the viewer (if exists) with the data
            if self.Dcmm.viewer != None: 
                self.Dcmm.viewer.sync()
        if self.render_mode == "depth_rgb_array":
            # Only keep the depth image
            imgs = imgs_depth
        return imgs

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None: self.Dcmm.viewer.close()

    def run_test(self):
        global trigger_delta, delta_x, delta_y, delta_z, trigger_delta_hand, delta_xyz_hand
        self.reset()
        action = np.zeros(16)
        while True:
            # Note: action's dim = 18, which includes 2 for the base, 4 for the arm, and 12 for the hand
            if trigger_delta:
                action[0:4] = np.array([delta_x, delta_y, delta_z, 0])
                trigger_delta = False
            else:
                action[0:4] = np.zeros(4)
            if trigger_delta_hand:
                print("delta_xyz_hand: ", delta_xyz_hand)
                action[4:16] = np.ones(12)*delta_xyz_hand
                trigger_delta_hand = False
            else:
                action[4:16] = np.zeros(12)
            arm_tensor = action[0:4]
            hand_tensor = action[4:16]

            actions_dict = {
                'arm': arm_tensor,
                'hand': hand_tensor,
            }
            observation, reward, terminated, truncated, info = self.step(actions_dict)
            # if terminated or truncated:
            #     print("terminated or truncated")
            #     break
def plot_one_joint(time_his,
                   joint_ctrl_his,
                   joint_his,
                   joint_target_his=None,
                   joint_id=1,          # 1~6
                   to_deg=True,
                   save_dir=None,
                   prefix="pid_joint",
                   show=True,
                   # —— 外观参数 —— 
                   fig_size=(9, 7),
                   title_size=20,
                   label_size=16,
                   tick_size=14,
                   legend_size=14,
                   line_width=2.2):
    """
    一次显示一个关节：
      上图：该关节控制量（tau）
      下图：该关节实际位置 q（实线）与目标位置 q_ref（虚线）
    支持配置字号、线宽、图尺寸。
    """
    # ---- 整理数据 ----
    t = np.asarray(time_his, dtype=float).reshape(-1)
    U = np.asarray(joint_ctrl_his, dtype=float)
    Q = np.asarray(joint_his, dtype=float)

    if t.size == 0 or U.size == 0 or Q.size == 0:
        raise ValueError("time_his / joint_ctrl_his / joint_his 为空，无法绘图")

    N = min(len(t), U.shape[0], Q.shape[0])
    t = t[:N]; U = U[:N]; Q = Q[:N]

    j = int(joint_id) - 1
    if j < 0 or j >= min(U.shape[1], Q.shape[1]):
        raise ValueError(f"joint_id 超界：传入 {joint_id}，但数据列数为 ctrl={U.shape[1]}、q={Q.shape[1]}")

    Qref = None
    if joint_target_his is not None and len(joint_target_his) > 0:
        Qref = np.asarray(joint_target_his, dtype=float)[:N]
        if Qref.shape[1] <= j:
            Qref = None

    q = Q[:, j]
    q_ref = Qref[:, j] if Qref is not None else None
    if to_deg:
        q = np.rad2deg(q)
        if q_ref is not None:
            q_ref = np.rad2deg(q_ref)
        y_label_q = "Position (deg)"
    else:
        y_label_q = "Position (rad)"

    tau = U[:, j]

    # ---- 绘图 ----
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=fig_size, sharex=True)
    ax1, ax2 = axes

    # 上：控制量
    ax1.plot(t, tau, label=f"tau{joint_id}", lw=line_width)
    ax1.set_ylabel("Torque / Control", fontsize=label_size)
    ax1.set_title(f"Joint {joint_id} Control & Position", fontsize=title_size)
    ax1.grid(True)
    ax1.legend(fontsize=legend_size)
    ax1.tick_params(axis='both', labelsize=tick_size)

    # 下：位置（实测 + 目标）
    ax2.plot(t, q, label=f"q{joint_id}", lw=line_width)
    if q_ref is not None:
        ax2.plot(t, q_ref, linestyle="--", label=f"q{joint_id}_ref", lw=line_width)
    ax2.set_xlabel("Time (s)", fontsize=label_size)
    ax2.set_ylabel(y_label_q, fontsize=label_size)
    ax2.grid(True)
    ax2.legend(fontsize=legend_size)
    ax2.tick_params(axis='both', labelsize=tick_size)

    plt.tight_layout()

    # 保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, f"{prefix}_j{joint_id}.png")
        fig.savefig(outpath, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
if __name__ == "__main__":
    os.chdir('../../')
    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    args = parser.parse_args()
    print("args: ", args)
    env = DcmmVecEnvArm(task='Catching', object_name='object', render_per_step=False, 
                    print_reward=False, print_info=False, 
                    print_contacts=False, print_ctrl=False, 
                    print_obs=False, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam, 
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test()
    # plot_one_joint(
    #     time_his=env.time_his,
    #     joint_ctrl_his=env.joint_ctrl_his,
    #     joint_his=env.joint_his,
    #     joint_target_his=env.joint_target_his,   # 没有就去掉
    #     joint_id=2,                # 指定关节：1~6
    #     to_deg=True,
    #     save_dir="./pid_plots",
    #     prefix="pid_bigfont",
    #     fig_size=(10, 8),
    #     title_size=22,
    #     label_size=18,
    #     tick_size=16,
    #     legend_size=16,
    #     line_width=2.5
    # )

