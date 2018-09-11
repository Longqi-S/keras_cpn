import os
import sys
import random
import math
import re
import time
import numpy as np
from models import cpn as cpn
import configs
from COCOAllJoints import COCOJoints
import tensorflow as tf
import argparse

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Training
############################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CPN.')
    parser.add_argument('--model', help="Path to weights .h5 file or 'coco'",
                        default='data/pretrain/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        type=str)
    parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    					default='configs/e2e_CPN_ResNet50_FPN_cfg.py',
                        type=str)
    parser.add_argument('--logs', required=False,
                        default=MODEL_DIR,
                        help='Logs and checkpoints directory (default=logs/)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config_file = os.path.basename(args.cfg.split('.')[0])
    config_def = eval('configs.' + config_file + '.Config')
    config = config_def()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs
    print("Model: ", args.model)
    print("Logs: ", args.logs)
    config.display()
    # Create model
    model = cpn.CPN(mode="training", config=config, model_dir=args.logs)  
    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    else:
        model_path = args.model
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)#, exclude=exclude)
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    tf.Session(config=config_tf)
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    coco_joints = COCOJoints()
    dataset_train, dataset_val = coco_joints.load_data(min_kps=1)

    # Training
    base_lr = config.LEARNING_RATE
    for i in range(0, 10):
        model.train(dataset_train, dataset_val,
                learning_rate=base_lr,
                epochs=10 * (i + 1),
                layers='all')
        base_lr = base_lr / 2