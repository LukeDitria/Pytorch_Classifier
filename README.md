# Pytorch_Classifier
General purpose Pytorch based image Classifier training code with transfer learning. <br>
Simply define your dataset and pick a classifier model from [Pytorch's Model zoo](https://pytorch.org/vision/0.13/models.html).<br>
Added option to also start from pre-trained weights by defining the weights string (something like IMAGENET1K_V1).<br>
Uses Pytorch AutoAugment with ImageNet Params. <br>
<b> Requires Pytorch 0.13+ </b>

## Training Examples
<b> Basic training command: </b><br>
This will create a ResNet18 model with random weights and will rescale images to 128x128.

```
python train.py -mn test_run --dataset_root #path to dataset root#
```

<b> General purpose training command: </b><br>
Good start for most things, this will create a ResNet50 model using the high-performance IMAGENET1K_V2 parameters and use 256x256 sized images

```
python train.py -mn resnet50_pretrained --dataset_root #path to dataset root# --model_type resnet50 --model_parameters IMAGENET1K_V2 --image_size 256
```

## Dataset
Training code uses the ImageFolder Pytorch Dataset and data should be organised as described in the docs. <br>
Training code assumes at least a train/test split has been defined, dataset_root parameter should point to the top level directory. <br>
[Example Dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)<br>
Folder structure should look something like the following <br>
```
├── dataset_root
    ├── train
    │   ├── class_0
    │   │   ├── img_0.png
    │   │   ├── ....png
    │   │   └── img_m.png
    │   ├── ....
    │   └── class_N
    │
    ├── test
    │   ├── class_0
    │   │   ├── img_0.png
    │   │   ├── ....png
    │   │   └── img_m.png
    │   ├── ....
    │   └── class_N
    │
    └── valid
        ├── class_0
        │   ├── img_0.png
        │   ├── ....png
        │   └── img_m.png
        ├── ....
        └── class_N

```
## To Do - requests welcome!
- Add additional dataloader for CSV type datasets
- Add dataset analysis
- Add learning rate decay
- Add additional loss function options
- Add early stopping for over-fitting
- Add code for example deployment
