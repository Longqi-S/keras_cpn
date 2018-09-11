# Cascaded Pyramid Network (CPN) based on keras (tensorflow backend)
## Results on COCO minival dataset (Single Model)
Note that our testing code is based on detection results from original tf-cpn (In COCO validation, detector AP is 41.1 whose human AP is 55.3).
<center>

| Method | Base Model | Input Size | AP @0.5:0.95 | AP @0.5 | AP @0.75 | AP medium | AP large |
|:-------|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| CPN | ResNet-50 | 384x288 | 71.1 | 88.9 | 77.7 | 67.2 | 78.0 |

</center>
We trained CPN model based on 4 GTX1080ti (11G) using 2days.

You can download our model here: https://github.com/Longqi-S/keras_cpn/releases/download/v0.1/cpn_resnet50_cpn_0065.h5

## Prepare

1. Download MSCOCO images from [http://cocodataset.org/#download](http://cocodataset.org/#download). We train in COCO [trainvalminusminival](https://drive.google.com/drive/folders/15loPFQCMQnJqLK1viSMeIwTFT-KbNzdG?usp=sharing) dataset and validate in [minival](https://drive.google.com/drive/folders/15loPFQCMQnJqLK1viSMeIwTFT-KbNzdG?usp=sharing) dataset. Then put the data and evaluation [PythonAPI](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI) in $CPN_ROOT/data/COCO/MSCOCO.

We use the human detection results same with tf-cpn, you can download it here: https://github.com/Longqi-S/keras_cpn/releases/download/v0.1/person_detection_minival411_human553.json.coco

After preparation, file stucture should be like below:
```
data/
       |->COCO/
       |    |->dets/
       |    |    |->person_detection_minival411_human553.json.coco
       |    |->MSCOCO/
       |    |    |->PythonAPI/
       |    |    |->train2014/
       |    |    |->val2014/
       |    |    |->person_keypoints_minival2014.json
       |    |    |->person_keypoints_trainvalminusminival2014.json
```

2. Download the base model (ResNet) weights from [keras model_zoo]
```
cd $CPN_ROOT/data
sh get_pretrain_model.sh
```

3. Setup your environment by first running
```
pip3 install -r requirement.txt
```

## Train

To train a CPN model, use train.py in root folder.
```
python3 train.py --model data/pretrain/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 --cfg configs/e2e_CPN_ResNet50_FPN_cfg.py
```
After the training finished, output is written underneath $CPN_ROOT/logs/ which looks like below
```
logs/
       |->resnet50_cpn20180819T1607/
       |    |->events.out.tfevents.1534666165.9507
       |    |->cpn_resnet50_cpn_0001.h5
       |    |->cpn_resnet50_cpn_0002.h5
       |    |->...
```

## Test
Run the testing code in the postprocessing folder. 
```
cd postprocessing
python3 mptest.py -d 0 -m cpn_resnet50_cpn_0002.h5 -c ../configs/e2e_CPN_ResNet50_FPN_cfg.py
```
### notice
We can only use just one GPU to test.

## How to draw network architecture

### Go to lib/utils/
```
python3 draw_net.py --mode 0 --cfg configs/e2e_CPN_ResNet50_FPN_cfg.py
```
mode: 0 means train; 1 means inference;
cfg : choose which network to draw;
