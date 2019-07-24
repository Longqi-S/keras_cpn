# --------------------------------------------------------------
# CPN: dataset preprocessing
# by Longqi-S
# --------------------------------------------------------------
import os
import numpy as np
import cv2
import random

""" three data augmentation choices
1) original img
2) fliped img
3) rotated img
"""
def data_augmentation(trainData, trainLabel, trainValids, cfg):
    tremNum = cfg.NR_AUG - 1
    gotData = trainData.copy()
    trainData   = np.append(trainData, \
            [trainData[0] for i in range(tremNum * len(trainData))], axis=0)
    trainLabel  = np.append(trainLabel, \
            [trainLabel[0] for i in range(tremNum * len(trainLabel))], axis=0)
    trainValids = np.append(trainValids, \
            [trainValids[0] for i in range(tremNum * len(trainValids))], axis=0)
    counter = len(gotData)
    for lab in range(len(gotData)):
        #ori_img       = gotData[lab].transpose(1, 2, 0)
        ori_img       = gotData[lab]
        annot         = trainLabel[lab].copy()
        annot_valid   = trainValids[lab].copy()
        height, width = ori_img.shape[0], ori_img.shape[1]
        center        = (width / 2., height / 2.)
        keypoints_num = cfg.KEYPOINTS_NUM
        affrat        = random.uniform(0.7, 1.35)
        halfl_w       = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h       = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img           = cv2.resize(ori_img[int(center[1] - halfl_h): \
                                           int(center[1] + halfl_h + 1),\
                                           int(center[0] - halfl_w): \
                                           int(center[0] + halfl_w + 1)], \
                                           (width, height))
        for i in range(keypoints_num):
            index1 = i << 1
            index2 = i << 1 | 1
            annot[index1] = (annot[index1] - center[0]) / halfl_w * (width - center[0]) + center[0]
            annot[index2] = (annot[index2] - center[1]) / halfl_h * (height - center[1]) + center[1]
            annot_valid[i] *= ((annot[index1] >= 0) & (annot[index1] < width) & \
                                (annot[index2] >= 0) & (annot[index2] < height))
        trainData[lab]   = img #img.transpose(2, 0, 1)
        trainLabel[lab]  = annot
        trainValids[lab] = annot_valid
        # flip augmentation
        newimg = cv2.flip(img, 1)
        cod  = []
        allc = []
        for i in range(keypoints_num):
            x, y = annot[i << 1], annot[i << 1 | 1]
            if x >= 0:
                x = width - 1 - x
            cod.append((x, y))
        trainData[counter] = newimg #newimg.transpose(2, 0, 1)
        # **** the joint index depends on the dataset ****
        for (q, w) in cfg.symmetry:
            cod[q], cod[w] = cod[w], cod[q]
        for i in range(keypoints_num):
            allc.append(cod[i][0])
            allc.append(cod[i][1])
        trainLabel[counter] = np.array(allc)
        allc_valid = annot_valid.copy()
        for (q, w) in cfg.symmetry:
            allc_valid[q], allc_valid[w] = allc_valid[w], allc_valid[q]
        trainValids[counter] = np.array(allc_valid)
        counter += 1
        # rotated augmentation
        for times in range(tremNum - 1):
            angle = random.uniform(0, 45)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            newimg = cv2.warpAffine(img, rotMat, (width, height))
            allc = []
            allc_valid = []
            for i in range(keypoints_num):
                x, y = annot[i << 1], annot[i << 1 | 1]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(coor[0])
                allc.append(coor[1])
                allc_valid.append(
                    annot_valid[i] * ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height)))
            #newimg = newimg.transpose(2, 0, 1)
            trainData[counter] = newimg
            trainLabel[counter] = np.array(allc)
            trainValids[counter] = np.array(allc_valid)
            counter += 1
        return trainData, trainLabel, trainValids

""" generate corresponding heatmaps
"""
def joints_heatmap_gen(data, label, cfg, return_valid=False, gaussian_kernel=(13, 13)):
    num_keypoints = cfg.KEYPOINTS_NUM
    tar_size      = cfg.OUTPUT_SHAPE
    ori_size      = cfg.DATA_SHAPE
    bat_size      = len(data)
    if return_valid:
        valid = np.ones((bat_size, num_keypoints), dtype=np.float32)
    ret = np.zeros((bat_size, num_keypoints, tar_size[0], tar_size[1]), dtype='float32')
    for i in range(bat_size):
        for j in range(num_keypoints):
            idx_x = j << 1; idx_y = j << 1 | 1
            if label[i][idx_x] < 0 or label[i][idx_y] < 0:
                continue
            label[i][idx_y] = min(label[i][idx_y], ori_size[0] - 1)
            label[i][idx_x] = min(label[i][idx_x], ori_size[1] - 1)
            ret[i][j][int(label[i][idx_y] * tar_size[0] / ori_size[0])][
                int(label[i][idx_x] * tar_size[1] / ori_size[1])] = 1
    for i in range(bat_size):
        for j in range(num_keypoints):
            ret[i, j] = cv2.GaussianBlur(ret[i, j], gaussian_kernel, 0)
    for i in range(bat_size):
        for j in range(num_keypoints):
            am = np.amax(ret[i][j])
            if am <= 1e-8:
                if return_valid:
                    valid[i][j] = 0.
                continue
            ret[i][j] /= am / 255
    if return_valid:
        return ret, valid
    else:
        return ret

def _preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""
    return (2.0 / 255.0) * inputs.astype(np.float32) - 1.0

""" image preprocessing
resnet from keras: just original img
resnet from tensorflow: 1) sub mean; 2) div 255;
"""
def image_preprocessing(inputs, config):
    img = inputs.astype(np.float32)
    if config.PIXEL_MEANS_VARS:
        img = img - config.PIXEL_MEANS
        if config.PIXEL_NORM:
            img = img / 255.
    return img

""" preprocessing method
1) read imgs and crop based on bbox;
2) data augment;
3) generate heatmaps;
"""
def preprocessing(d, config, stage='train', debug=False):
    height, width = config.DATA_SHAPE
    imgs   = []
    labels = []
    valids = []
    # read image
    img_ori = cv2.imread(d['imgpath'])
    if img_ori is None:
        print('read none image')
        return None
    # crop based on bbox
    img = img_ori.copy()
    add = max(img.shape[0], img.shape[1])
    ## make border and avoid crop size exceed image size
    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,\
            value=config.PIXEL_MEANS.reshape(-1))
    bbox = np.array(d['bbox']).reshape(4, ).astype(np.float32)
    ## coordinate should also translate of 'add'
    bbox[:2] += add
    if 'joints' in d:
        joints = np.array(d['joints']).reshape(config.KEYPOINTS_NUM, 3).astype(np.float32)
        joints[:, :2] += add
        idx = np.where(joints[:, -1] == 0)
        joints[idx, :2] = -1000000
    ## preprare crop size
    crop_width  = bbox[2] * (1 + config.imgExtXBorder * 2)
    crop_height = bbox[3] * (1 + config.imgExtYBorder * 2)
    objcenter = np.array([bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.])
    ## in training stage, crop size should extend 0.25
    if stage == 'train':
        crop_width  = crop_width * (1 + 0.25)
        crop_height = crop_height * (1 + 0.25)

    if crop_height / height > crop_width / width:
        crop_size = crop_height
        min_shape = height
    else:
        crop_size = crop_width
        min_shape = width
    crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
    crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)
    min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
    max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
    min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
    max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)
    x_ratio = float(width) / (max_x - min_x)
    y_ratio = float(height) / (max_y - min_y)
    if 'joints' in d:
        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        joints[:, 0] *= x_ratio
        joints[:, 1] *= y_ratio
        label = joints[:, :2].copy()
        valid = joints[:, 2].copy()
    img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
    if stage != 'train':
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add])
    if debug:
        from visualization import draw_skeleton
        img2 = img.copy()
        draw_skeleton(img2, joints.astype(int))
        cv2.imshow('', img2)
        cv2.imshow('1', img_ori)
        cv2.waitKey()
    if 'joints' in d:
        labels.append(label.reshape(-1))
        valids.append(valid.reshape(-1))
    imgs.append(img)
    if stage == 'train':
        imgs, labels, valids = data_augmentation(imgs, labels, valids, config)
        heatmaps15 = joints_heatmap_gen(imgs, labels, config, return_valid=False, \
                    gaussian_kernel=config.GK15)
        heatmaps11 = joints_heatmap_gen(imgs, labels, config, return_valid=False, \
                    gaussian_kernel=config.GK11)
        heatmaps9  = joints_heatmap_gen(imgs, labels, config, return_valid=False, \
                    gaussian_kernel=config.GK9)
        heatmaps7  = joints_heatmap_gen(imgs, labels, config, return_valid=False, \
                    gaussian_kernel=config.GK7)
        imgs = imgs.astype(np.float32)
        for index_ in range(len(imgs)):
            imgs[index_] = image_preprocessing(imgs[index_], config)

        return_args = [imgs.astype(np.float32),
            heatmaps15.astype(np.float32).transpose(0, 2, 3, 1),
            heatmaps11.astype(np.float32).transpose(0, 2, 3, 1),
            heatmaps9.astype(np.float32).transpose(0, 2, 3, 1),
            heatmaps7.astype(np.float32).transpose(0, 2, 3, 1),
            valids.astype(np.float32)]

        return return_args
    else:
        for index_ in range(len(imgs)):
            imgs[index_] = image_preprocessing(imgs[index_], config)
        return [np.asarray(imgs).astype(np.float32), details]

if __name__ == '__main__':
    import sys
    sys.path.append('../data/COCO')
    sys.path.append('../lib/utils')
    sys.path.append('../models')
    from COCOAllJoints import COCOJoints
    coco_joints = COCOJoints()
    train, _ = coco_joints.load_data(min_kps=1)
    from models.config import DefaultConfig
    config = DefaultConfig()
    data = preprocessing(train[0], config, stage='train', debug=False)
    from IPython import embed; embed()
