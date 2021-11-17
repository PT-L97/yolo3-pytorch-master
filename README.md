## yolo3-pytorch-master

## Directory

1. [Performance](#Performance)
2. [Environment](#Environment)
3. [Download](#Data_Download)
4. [How2Train](#Train_step)
5. [How2eval](#Evaluate_Step)
6. [Result](#Result)
7. [Reference](#Reference)

## Performance

| Train_dataset |                       weight file name                       | Test_dataset | picture size | mAP 0.5:0.95 | mAP 0.5 |
| :-----------: | :----------------------------------------------------------: | :----------: | :----------: | :----------: | :-----: |
| SIXray_Train  | [ep100-loss3.098-val_loss3.129.pth](https://github.com/PT-L97/yolo3-pytorch-master/blob/a63f6ab2a27d8517523bd53ec4c17545a7bf85fa/yolo-pytorch-master/logs/ep100-loss3.098-val_loss3.129.pth) | SIXray_test  |   416x416    |     38.0     |  67.2   |


The SIXray weight is training based on [yolo_weights.pth](https://github.com/bubbliiiing/yolo3-pytorch/releases/download/v1.0/yolo_weights.pth)
## Environment

torch == 1.7

cuda == 10  

cudnn == 7.4.1.5

for more details : requirement.txt

## Data_Download

SIXray dataset: 

link：https://pan.baidu.com/s/1wxwY5orhZ2ugc5gV4i4ZTg 
extract code：8aay 

This paper would use Pascal VOC XML Annotation Format.

## Train_step

1. prepare basic data
   extract zip to VOCdevkit

2. get *.xml from negative samples(non-object)

  create_xml.py

   modify src_img_dir = "negative_sample/0" and src_xml_dir = "negative_sample/XML"

3. split dataset   
   modify voc_annotation.py: annotation_mode=0

  run voc_annotation.py 

  get root directory: 2007_train.txt and 2007_val.txt

  get *.txt in yolo3-pytorch-master/VOCdevkit/VOC2007/ImageSets/Main

4. train  
   modify model_path and classes_path   

  run train.py

## Evaluate_Step

1. using *.xml to evaluate 

2. modify voc_annotation.py: trainval_percent --for basic code,  (train + val) : test = 9:1 

   modify voc_annotation.py: train_percent -- for basic code, train: val = 9:1 

3. modify model_path and classes_path

   run yolo.py

4. run get_map.py to get evaluation result in directory ./map_out

5. to get 0.50:0.95map, install pycocotools. 

   for more details: pycocotools_on_Ubuntu.txt

## Result

<img src="https://github.com/PT-L97/yolo3-pytorch-master/blob/a63f6ab2a27d8517523bd53ec4c17545a7bf85fa/yolo-pytorch-master/map_out/results/mAP.png" width="800"/>

for more details: map_out

## Reference

https://github.com/MeioJane/SIXray

https://github.com/qqwweee/keras-yolo3  

https://github.com/eriklindernoren/PyTorch-YOLOv3   

https://github.com/BobLiu20/YOLOv3_PyTorch

https://github.com/cocodataset/cocoapi
