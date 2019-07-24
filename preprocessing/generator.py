# --------------------------------------------------------------
# CPN: data generator
# by Longqi-S
# --------------------------------------------------------------
import numpy as np
from preprocessing.dataset import preprocessing
import logging
"""data generator
used for training and validation
"""
def data_generator(dataset, config, mode='train', shuffle=True, batch_size=1):

    b = 0  # batch item index
    image_index = -1
    total_size = len(dataset)
    image_ids = [i for i in range(total_size)]
    error_count = 0
    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % total_size
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)
            image_id = image_ids[image_index]
            data_item = dataset[image_id]
            data_res = preprocessing(data_item, config, stage=mode)
            imgs = data_res[0]
            heatmaps15 = data_res[1]
            heatmaps11 = data_res[2]
            heatmaps9  = data_res[3]
            heatmaps7  = data_res[4]
            valids     = data_res[5]
            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + imgs.shape[1:], dtype=imgs.dtype)
                batch_heatmaps15 = np.zeros(
                    (batch_size,) + heatmaps15.shape[1:], dtype=heatmaps15.dtype)
                batch_heatmaps11 = np.zeros(
                    (batch_size,) + heatmaps11.shape[1:], dtype=heatmaps11.dtype)
                batch_heatmaps9  = np.zeros(
                    (batch_size,) + heatmaps9.shape[1:], dtype=heatmaps9.dtype)
                batch_heatmaps7 = np.zeros(
                    (batch_size,) + heatmaps7.shape[1:], dtype=heatmaps7.dtype)
                batch_valids = np.zeros(
                    [batch_size, config.KEYPOINTS_NUM], dtype=valids.dtype)


            # Add to batch
            augment_num = config.NR_AUG
            batch_images[b: b + augment_num]     = imgs
            batch_heatmaps15[b: b + augment_num] = heatmaps15
            batch_heatmaps11[b: b + augment_num] = heatmaps11
            batch_heatmaps9[b: b + augment_num]  = heatmaps9
            batch_heatmaps7[b: b + augment_num]  = heatmaps7
            batch_valids[b: b + augment_num]     = valids
            b += augment_num

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_heatmaps15, batch_heatmaps11, batch_heatmaps9,
                          batch_heatmaps7, batch_valids]
                outputs = []

                yield inputs, outputs ## Similar to return; begin from next line
                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset[image_id]))
            error_count += 1
            if error_count > 5:
                raise
