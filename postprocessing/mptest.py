import os
import os.path as osp
import sys
sys.path.append('../')
import numpy as np
import argparse
import cv2
import time
import configs
from preprocessing.dataset import preprocessing
from models import cpn as modellib

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

import keras.backend as K

def test_net(model, dets, ranges, config):
    min_scores = 1e-10
    min_box_size = 0.  # 8 ** 2
    all_res = []
    dump_results = []
    N = len(ranges)
    start_time = time.time()
    for index_ in range(N - 1):
        det_range = [ranges[index_], ranges[index_ + 1]]
        img_start = det_range[0]
        while img_start < det_range[1]:
            img_end = img_start + 1
            im_info = dets[img_start]
            while img_end < det_range[1] and dets[img_end]['image_id'] == im_info['image_id']:
                img_end += 1
            test_data = dets[img_start: img_end]
            img_start = img_end
            iter_avg_cost_time = (time.time() - start_time) / (img_end - ranges[0])
            print('ran %.ds >> << left %.ds' % (
                iter_avg_cost_time * (img_end - ranges[0]), iter_avg_cost_time * (ranges[-1] - img_end)))
            all_res.append([])
            # get box detections
            cls_dets = np.zeros((len(test_data), 5), dtype=np.float32)
            for i in range(len(test_data)):
                bbox = np.asarray(test_data[i]['bbox'])
                cls_dets[i, :4] = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                cls_dets[i, 4] = np.array(test_data[i]['score'])
            # nms and filter
            keep = np.where((cls_dets[:, 4] >= min_scores) &
                        ((cls_dets[:, 3] - cls_dets[:, 1]) * (cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
            cls_dets = cls_dets[keep]
            test_data = np.asarray(test_data)[keep]
            if len(keep) == 0:
                continue
            # crop and detect keypoints
            cls_skeleton = np.zeros((len(test_data), config.KEYPOINTS_NUM, 3))
            crops = np.zeros((len(test_data), 4))
            batch_size = 8
            for test_id in range(0, len(test_data), batch_size):
                start_id = test_id
                end_id = min(len(test_data), test_id + batch_size)
                test_imgs = []
                details = []
                for ii_ in range(start_id, end_id):
                    res_tmp = preprocessing(test_data[ii_], config, stage='test')
                    test_imgs.append(res_tmp[0])
                    details.append(res_tmp[1])
                details = np.asarray(details)
                feed = test_imgs
                for ii_ in range(end_id - start_id):
                    ori_img = test_imgs[ii_][0]
                    flip_img = cv2.flip(ori_img, 1)
                    feed.append(flip_img[np.newaxis, ...])
                feed = np.vstack(feed)
                
                ## model predict
                res = model.keras_model.predict([feed], verbose=0)
                res = res.transpose(0, 3, 1, 2) # [batch, kps, h, w]
                ## combine flip result
                for ii_ in range(end_id - start_id):
                    fmp = res[end_id - start_id + ii_].transpose((1, 2, 0))
                    fmp = cv2.flip(fmp, 1)
                    fmp = list(fmp.transpose((2, 0, 1)))
                    for (q, w) in config.symmetry:
                        fmp[q], fmp[w] = fmp[w], fmp[q]
                    fmp = np.array(fmp)
                    res[ii_] += fmp
                    res[ii_] /= 2
                for test_image_id in range(start_id, end_id):
                    r0 = res[test_image_id - start_id].copy()
                    r0 /= 255
                    r0 += 0.5 # ??
                    for w in range(config.KEYPOINTS_NUM):
                        res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])
                    border = 10
                    dr = np.zeros((config.KEYPOINTS_NUM, config.OUTPUT_SHAPE[0] + 2 * border, config.OUTPUT_SHAPE[1] + 2 * border))
                    dr[:, border:-border, border:-border] = res[test_image_id - start_id][: config.KEYPOINTS_NUM].copy()
                    for w in range(config.KEYPOINTS_NUM):
                        dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)
                    ## find first max and second max one
                    for w in range(config.KEYPOINTS_NUM):
                        lb = dr[w].argmax()
                        y, x = np.unravel_index(lb, dr[w].shape)
                        dr[w, y, x] = 0
                        lb = dr[w].argmax()
                        py, px = np.unravel_index(lb, dr[w].shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, config.OUTPUT_SHAPE[1] - 1))
                        y = max(0, min(y, config.OUTPUT_SHAPE[0] - 1))
                        cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2) ## why add 2?
                        cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]
                    # map back to original images
                    crops[test_image_id, :] = details[test_image_id - start_id, :]
                    for w in range(config.KEYPOINTS_NUM):
                        cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / config.DATA_SHAPE[1] * (
                        crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
                        cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / config.DATA_SHAPE[0] * (
                        crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
            all_res[-1] = [cls_skeleton.copy(), cls_dets.copy()]
            cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, config.KEYPOINTS_NUM)
            cls_skeleton[:, :, 2] = 1
            cls_scores = cls_dets[:, -1].copy()
            # rescore
            cls_dets[:, -1] = cls_scores * cls_partsco.mean(axis=1)
            cls_skeleton = np.concatenate(
                [cls_skeleton.reshape(-1, config.KEYPOINTS_NUM * 3), (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
                axis=1)
            for i in range(len(cls_skeleton)):
                result = dict(image_id=im_info['image_id'], category_id=1, score=float(round(cls_skeleton[i][-1], 4)),
                          keypoints=cls_skeleton[i][:-1].round(3).tolist())
                dump_results.append(result)
    return all_res, dump_results

def test(test_model, dets_path, gt_path):
    # loading model
    """
    config_file = os.path.basename(args.cfg).split('.')[0]
    config_def = eval('configs.' + config_file + '.Config')
    config = config_def()
    config.GPUs = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs

    model = modellib.CPN(mode="inference", config=config, model_dir="")
    model.load_weights(test_model, by_name=True)
    """
    eval_gt = COCO(gt_path)
    """
    import json
    with open(dets_path, 'r') as f:
        dets = json.load(f)
    dets = [i for i in dets if i['image_id'] in eval_gt.imgs]
    dets = [i for i in dets if i['category_id'] == 1]
    dets.sort(key=lambda x: (x['image_id'], x['score']), reverse=True)
    for i in dets:
        i['imgpath'] = '../data/COCO/MSCOCO/val2014/COCO_val2014_000000%06d.jpg' % i['image_id']
    img_num = len(np.unique([i['image_id'] for i in dets]))
    #from IPython import embed; embed()
    ranges = [0]
    img_start = 0
    for run_img in range(img_num):
        img_end = img_start + 1
        while img_end < len(dets) and dets[img_end]['image_id'] == dets[img_start]['image_id']:
            img_end += 1
        if (run_img + 1) % config.IMAGES_PER_GPU == 0 or (run_img + 1) == img_num:
            ranges.append(img_end)
        img_start = img_end
    _, dump_results = test_net(model, dets, ranges, config)
    output_dir = "logs"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    result_path = osp.join(output_dir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(dump_results, f)
    """
    result_path = osp.join("logs", 'results.json')
    eval_dt = eval_gt.loadRes(result_path)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    K.clear_session()
        
    
if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', '-d', type=str, dest='gpu_id')
        parser.add_argument('--model', '-m', required=True, type=str, dest='test_model')
        parser.add_argument('--cfg', '-c', required=True, type=str, dest='cfg')
        args = parser.parse_args()
        print(args.test_model)
        return args
    global args
    args = parse_args()
    dets_path = "../data/COCO/dets/person_detection_minival411_human553.json.coco"
    gt_path  = "../data/COCO/MSCOCO/person_keypoints_minival2014.json"
    if args.test_model:
        test(args.test_model, dets_path, gt_path)
    #from IPython import embed; embed()

