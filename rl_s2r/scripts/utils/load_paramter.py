import yaml
import numpy as np

def load_camera_config(config_file):
    """从YAML文件加载相机内参和旋转矩阵"""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # 解析相机内参和旋转矩阵
    K_color = np.array(config['K_color'])
    K_depth = np.array(config['K_depth'])
    R_color_to_depth = np.array(config['R_color_to_depth'])
    
    return K_color, K_depth, R_color_to_depth

