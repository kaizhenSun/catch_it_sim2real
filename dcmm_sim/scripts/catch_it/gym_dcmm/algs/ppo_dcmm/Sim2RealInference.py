import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import time

from .utils import RunningMeanStd
from .models_catch import ActorCritic

class Sim2RealInference:
    def __init__(self, env=None):
        """

        """
        # Parameters
        self.device = "cuda:0" # 'cuda:?', -1 for 'cpu'
        self.obs_t_shape = (16,)
        self.normalize_input = True
        self.full_action_dim = 16
        self.task = 'Catching'
        self.action_track_denorm = np.array([1.5, 0.025, 0.01])
        self.action_catch_denorm = np.array([1.5, 0.025, 0.15])
        self.env = env
        # ---- Model ----
        net_config = {
            'actor_units': [256, 128],
            'actions_num': 16,
            'input_shape': (28,),
            'separate_value_mlp': True,
        }
        print("net_config: ", net_config)
        self.model = ActorCritic(net_config)
        self.model.to(self.device)

        self.running_mean_std_track = RunningMeanStd(self.obs_t_shape).to(self.device)
        self.running_mean_std_hand = RunningMeanStd((12,)).to(self.device)

        self.obs = None

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location = self.device)
        if self.normalize_input:
            self.running_mean_std_track.load_state_dict(checkpoint['running_mean_std_track'])
            self.running_mean_std_hand.load_state_dict(checkpoint['running_mean_std_hand'])
        if not fn:
            return
        self.model.load_state_dict(checkpoint['model'])

    def obs2tensor(self, obs, task='Catching'):

        # Map the step result to tensor
        if self.task == 'Catching':
            obs_array = np.concatenate((
                        # obs["base"]["v_lin_2d"], 
                        obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], 
                        obs["arm"]["ee_v_lin_3d"],
                        obs["object"]["pos3d"], obs["object"]["v_lin_3d"], 
                        obs["hand"],
                        ), axis=1)
        else:
            obs_array = np.concatenate((
                    # obs["base"]["v_lin_2d"], 
                    obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], 
                    obs["arm"]["ee_v_lin_3d"], 
                    obs["object"]["pos3d"], obs["object"]["v_lin_3d"]
                    ), axis=1)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        return obs_tensor

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        # De-normalize the actions
        if self.task == 'Tracking':
            # base_tensor = actions[:, :2] * self.action_track_denorm[0]
            arm_tensor = actions[:, :4] * self.action_track_denorm[1]
            hand_tensor = actions[:, 4:] * self.action_track_denorm[2]
        else:
            # base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, :4] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 4:] * self.action_catch_denorm[2]
        actions_dict = {
            'arm': arm_tensor,
            # 'base': base_tensor,
            'hand': hand_tensor
        }
        return actions_dict
    
    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std_track.eval()
            self.running_mean_std_hand.eval()

    def model_act(self, obs_dict):
        processed_obs_track = self.running_mean_std_track(obs_dict['obs'][:, :-12])
        processed_obs_hand = self.running_mean_std_hand(obs_dict['obs'][:, -12:])
        processed_obs = torch.cat((processed_obs_track, processed_obs_hand), dim=1)
        input_dict = {
            'obs': processed_obs,
            'obs_t': processed_obs[:,:-12],
            'obs_c': processed_obs[:,:],
        }

        res_dict = {}
        res_dict['actions'] = self.model.act_inference(input_dict)
        return res_dict

    def predict(self, obs):
        obs = {'obs': self.obs2tensor(obs)}
        res_dict = self.model_act(obs)
        actions = res_dict['actions']
        actions[:,:] = torch.clamp(actions[:,:], -1, 1)
        actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
        actions_dict = self.action2dict(actions)
        return actions_dict
    
#     def play_test_steps(self):
#         # start_time = time.time()
#         res_dict = self.model_act(self.obs)
#         # print("time: ", time.time() - start_time)
#         actions = res_dict['actions']
#         actions[:,:] = torch.clamp(actions[:,:], -1, 1)
#         actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
#         actions_dict = self.action2dict(actions)
#         return actions_dict


#     def play_test_env_steps(self):
#         start_time = time.time()
#         res_dict = self.model_act(self.obs)
#         print("time: ", time.time() - start_time)
#         actions = res_dict['actions']
#         actions[:,:] = torch.clamp(actions[:,:], -1, 1)
#         actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
#         actions_dict = self.action2dict(actions)
#         # print("actions_dict: ", actions_dict)
#         obs, r, terminates, truncates, infos = self.env.step(actions_dict)
#         self.obs = {'obs': self.obs2tensor(obs)}

#     def test_env(self):
#         self.set_eval()
#         reset_obs, _ = self.env.reset()
#         self.obs = {'obs': self.obs2tensor(reset_obs)}
#         while(True):
#             self.play_test_env_steps()
    
#     def test(self):
#         self.set_eval()
#         # reset_obs, _ = self.env.reset()
#         reset_obs = OrderedDict([
#             ('arm', OrderedDict([
#                 ('ee_pos3d', np.array([[0.17215596, 0.01894671, 0.26021287]], dtype=np.float32)),
#                 ('ee_quat', np.array([[-0.59072244, 0.39029247, -0.3971115, 0.58453584]], dtype=np.float32)),
#                 ('ee_v_lin_3d', np.array([[-0.00069342, -0.00075518, 0.00286768]], dtype=np.float32)),
#                 ('joint_pos', np.array([[0.21672745, 1.1046853, -0.22331546, 0.2191322, -1.1983758, -0.15225583]], dtype=np.float32))
#             ])),
#             ('hand', np.array([[2.3520093, -0.10279789, 0.9765025, 1.8407148, -0.54845285, 
#                                 0.01487582, 2.341205, -0.49063003, -0.21091968, -0.66344714, 
#                                 -1.2759252, -1.3659493]], dtype=np.float32)),
#             ('object', OrderedDict([
#                 ('pos3d', np.array([[0.22188169, 0.01739461, 0.30470577]], dtype=np.float32)),
#                 ('v_lin_3d', np.array([[-0.01986971, -0.01096101, 0.05682832]], dtype=np.float32))
#             ]))
#         ])


#         self.obs = {'obs': self.obs2tensor(reset_obs)}
#         while(True):
#             _ = self.play_test_steps()

# if __name__ == '__main__':

#     catch_pt_path = '/home/kaizhen/rl_ws/sim2real_ws/src/rl_s2r/scripts/inference/models/catch_two_stage1.pth'
#     env = None
#     inference = Sim2RealInference(env)
#     inference.restore_test(catch_pt_path)
#     inference.test()
    