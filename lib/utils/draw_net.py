# --------------------------------------------------------------
# CPN: Draw network
# by Longqi-S
# --------------------------------------------------------------
import sys
import os
sys.path.append('../../')
import time
import numpy as np

from models.config import DefaultConfig 
from models import cpn as modellib
import cv2
import tensorflow as tf
import argparse
import configs

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

""" parse args
mode: 0-> train ... 1-> infer
cfg : determine which architecture to draw
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=0, type=int,
                        help="train or inference mode")
    parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    					default='configs/e2e_CPN_ResNet50_FPN_cfg.py',
                        type=str)
    return parser.parse_args()

def draw_net():
    args = parse_args()
    config_file = args.cfg
    #from IPython import embed; embed()
    config_def = eval('configs.' + os.path.basename(config_file.split('.')[0]) + '.Config')
    config = config_def()
    modes = ["training", "inference"]
    choice = args.mode
    config.GPUs = '0'
    model = modellib.CPN(mode=modes[choice], config=config, model_dir=MODEL_DIR)
    from keras.utils import plot_model
    plot_model(model.keras_model, to_file='network_arch_' + str(modes[choice]) + '.png', show_shapes=True)

    import keras.backend as K
    K.clear_session()

if __name__ == '__main__':
    draw_net()
