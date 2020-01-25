# SAR_River_Segmentation_Pytorch-Unet

![Alt text](./readmeFigure/result.jpg?raw=true "Mask prediction exemple")

U-net architecture for river segmentation from Synthetic Aperture Radar (SAR) images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them

```
matplotlib
numpy
Pillow
torch
torchvision
tensorboard
pydensecrf
scikit-learn
```

### Installing

```
git clone https://github.com/AyoubOuddah/SAR_River_Segmentation_Pytorch-Unet.git
```
```shell script
Cloning into 'SAR_River_Segmentation_Pytorch-Unet'...
remote: Enumerating objects: 144, done.
remote: Counting objects: 100% (144/144), done.
remote: Compressing objects: 100% (104/104), done.
remote: Total 144 (delta 85), reused 87 (delta 36), pack-reused 0
Receiving objects: 100% (144/144), 3.07 MiB | 1.92 MiB/s, done.
Resolving deltas: 100% (85/85), done.
```
### Training

```shell script
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]
                [-W WEIGHTS [WEIGHTS ...]]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 10.0)
  -W WEIGHTS [WEIGHTS ...], --weights WEIGHTS [WEIGHTS ...]
                        wights used for the imbalenced data (default: [1])
```
By default, the `scale` is 1, so if you wish to obtain better performances or use less memory you can reduce it's value.
The input patch should be in the `data/dataset/train`. Each patch must be in this format a numpyArray of 3 channels contrains [VV, VH, GT] the patches can be generated using the 2 scripts `utils/sar_dataset.py` to extract, normalize and compute wights for imbalanced datasets and `utils/generate_patches.py` to generate the patchs. 
To resume training from a model use `--load path_to_model.pth` for exemple you have a model trained for 15 Epochs and you want to train it for another 15 Epochs in this case you need to use `--epochs 30` rather then `--epochs 15` 

### Prediction

You can easily test the output masks on your images via the CLI.

To predict a single image and save it:
```shell script
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]
                  [--chennel CHENNEL]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 1)
  --chennel CHENNEL, -c CHENNEL
                        number of channels in the patch (default: 1)
```

### Compute the performances on the testset 

```shell script
usage: evalTop1.py [-h] [--model FILE] [--mask-threshold MASK_THRESHOLD]
                   [--scale SCALE] [--chennel CHENNEL] [-b [B]]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 1)
  --chennel CHENNEL, -c CHENNEL
                        number of channels in the patch (default: 1)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
```

Computing some metrics on the testset (Dice coefficient, Jaccard, Accuracy, Pres, Recall, F1_score)
output exemple :

```shell script
Dice coefficient :  0.5218227899955027
Jaccard:  0.4044690628804068
Accuracy:  0.9419250712137516
Pres : 0.6784951292794088
Recall:  0.4833312274225643
F1_score  0.5645213361586173
```

## Tensorboard
You can visualize in real time the train and test losses, along with the model predictions and some useful metrics with tensorboard:

`tensorboard --logdir=runs`

## Runing on Google Colab

the folder `src/Google-Colab` contains a set of notebooks that can easily used to generate the patchs, train the model, predict and evaluate the models performances.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* @milesial (Pytorch-UNet)
* Emanuele Dalsasso (Script for generating the patchs) 
* Nicolas GASNIER (SAR dataset)  
* Billie Thompson @PurpleBooth (readme template) 
