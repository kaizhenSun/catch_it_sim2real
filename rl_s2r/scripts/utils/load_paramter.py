import yaml
import numpy as np

def load_config(config_file, keys):
    """
    从YAML文件加载指定的参数
    
    参数:
    config_file: str, YAML配置文件的路径
    keys: dict, 参数的键值对，键为参数名，值为目标类型或结构，例如 `np.array` 或 `list`
    
    返回:
    dict, 参数名和加载的值
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    loaded_params = {}

    for key, expected_type in keys.items():
        if key in config:
            value = config[key]
            
            # 根据预期类型转换数据
            if expected_type == np.array:
                loaded_params[key] = np.array(value)
            elif expected_type == list:
                loaded_params[key] = list(value)
            elif expected_type == float:
                loaded_params[key] = float(value)
            elif expected_type == int:
                loaded_params[key] = int(value)
            else:
                # 如果遇到未知类型，直接返回原数据
                loaded_params[key] = value
        else:
            print(f"Warning: Key '{key}' not found in config file.")
            loaded_params[key] = None

    return loaded_params

def load_camera_config(config_file):
    """从YAML文件加载相机内参和旋转矩阵"""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    K_color = np.array(config['K_color'])
    K_depth = np.array(config['K_depth'])
    R_color_to_depth = np.array(config['R_color_to_depth'])
    
    return K_color, K_depth, R_color_to_depth
