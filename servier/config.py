import os

from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')
config = OmegaConf.load(config_path)
