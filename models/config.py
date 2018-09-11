# --------------------------------------------------------------
# CPN: default configuration
# by Longqi-S
# --------------------------------------------------------------

"""CPN config system.
This file specifies default config options for CPN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.
Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See configs/*/*.yaml for example config files
CPN supports a lot of different backbone or model, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import math
import numpy as np

class DefaultConfig(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    NAME = "CPN"  # Override in sub-classes
    #########################################################################
    ## training configuration
    #########################################################################
    GPUs = '0'
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 24 384x288px.
    IMAGES_PER_GPU = 24
    STEPS_PER_EPOCH = 1000
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50
    KEYPOINTS_NUM = 17
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 1e-5
    LEARNING_OPTIMIZER = 'SGD' ## 'SGD', 'ADAM'
    GRADIENT_CLIP_NORM = 5.0
    #########################################################################
    ## preprocessing configuration
    #########################################################################
    DATA_SHAPE = (384, 288) #height, width
    OUTPUT_SHAPE = (96, 72) #height, width
    GAUSSAIN_KERNEL = (13, 13)
    #
    GK15 = (23, 23)
    GK11 = (17, 17)
    GK9 = (13, 13)
    GK7 = (9, 9)
    DATA_AUG = True # has to be true
    NR_AUG = 4
    PIXEL_MEANS_VARS = False
    PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]]) # BGR
    PIXEL_NORM = True
    imgExtXBorder = 0.1
    imgExtYBorder = 0.15
    symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    #########################################################################
    ## network configuration
    #########################################################################
    BACKBONE = 'resnet50' 
    
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        num_gpus = len(self.GPUs.split(','))
        self.BATCH_SIZE = self.IMAGES_PER_GPU * num_gpus

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
import sys
import os
import os.path as osp
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(osp.abspath(osp.join(cur_dir, '../data/COCO')))
sys.path.append(osp.abspath(osp.join(cur_dir, '../data/COCO/MSCOCO/PythonAPI')))
