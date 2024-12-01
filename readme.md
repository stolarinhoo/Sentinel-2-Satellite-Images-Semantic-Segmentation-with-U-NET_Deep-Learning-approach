# Sentinel-2 Satellite Images Semantic Segmentation with U-NET: A Deep Learning Approach

Welcome to this repository, which contains the complete codebase for training and evaluating a U-NET model designed for semantic segmentation of Sentinel-2 satellite imagery. This project harnesses the power of deep learning to analyze and classify satellite images, with a particular focus on understanding land cover changes in and around urban areas. 

The deep learning model was trained with 1,728 training samples and validated with 172 samples. The model achieved a precision of approximately 75% for correctly predicted classes, demonstrating its effectiveness in classifying land cover from satellite images.


## Environment and Library Installation (conda)

To run this project, it's recommended to set up a dedicated conda environment. You can do so with the following commands:

```bash
conda create -n sentinel2_unet python=3.9.16
conda activate sentinel2_unet
```

After activating the environment, you can install the necessary libraries with:

```bash
conda install --file requirements.txt
```

## Sample Usage
```bash
python main.py \
    --images-dir ./data/images \
    --labels-dir ./data/labels \
    --train-size 0.8 \
    --batch-size 32 \
    --epochs 100 \
    --save-model \
    --plot-images \
    --num-samples 5 \
    --save-samples \
    --save-training-summary
```

## Base Satellite Image

Sentinel-2 is an Earth observation satellite providing high-resolution images. The base image used for learning and validation was captured by the Sentinel-2B satellite on 7 June 2018: 
```
S2B_MSIL2A_20180607T095029_N0208_R079_T33UYS_20180607T130225
```

This image covers the southwestern part of Poland, including parts of the Lower Silesian and Opole Voivodships. It encompasses the central and eastern parts of Wrocław and the western part of Brzeg.

![img1](/docs/img1.jpg)

## Classes

The focus of the work was on urban areas, so the land cover classes were expanded to include more urbanized land cover. The model is trained to identify 8 classes:
| L.P. | Land Cover Class                             |
| ---- | -------------------------------------------- |
| 1    | Agricultural areas                           |
| 2    | Forest and semi naturals                     |
| 3    | Wetlands                                     |
| 4    | Water bodies                                 |
| 5    | Urban fabric                                 |
| 6    | Industrial, commercial and transport units   |
| 7    | Mine, dump and construction sites            |
| 8    | Artificial, non-agricultural vegetated areas |


## Data Preparation for Training

The data was extracted from the base image. First, all bands were resampled to a spatial resolution of 10m and then the image was divided into 256 x 256 tiles with labels. The labels cover the area of the central and eastern part of Wrocław and the western part of Brzeg. A total of 1,920 learning samples were obtained.

Shapes:
- Real image: 256 x 256 x 12
- Label: 256 x 256 x 1

![img2](/docs/img2.png)

A sample of learning samples can be found in the data directory.

## Training 

Training was conducted with 1,728 training samples and 172 validation samples. The number of epochs was set to 60. The model achieved a precision of approximately 75% for correctly predicted classes.

Here are the results of the last five epochs:
| Epoch   | Loss    | Accuracy | Val_Loss | Val_Accuracy | LR        |
|---------|---------|----------|----------|--------------|-----------|
| 56/60   | 0.1762  | 0.7461   | 0.2021   | 0.7590       | 1.0000e-04|
| 57/60   | 0.1741  | 0.7465   | 0.1999   | 0.7601       | 1.0000e-04|
| 58/60   | 0.1720  | 0.7470   | 0.1980   | 0.7613       | 1.0000e-04|
| 59/60   | 0.1699  | 0.7475   | 0.1961   | 0.7625       | 1.0000e-04|
| 60/60   | 0.1678  | 0.7480   | 0.1943   | 0.7638       | 1.0000e-04|

## Training Validation
![img3](/docs/img3.png)

Model Accuracy:
- Model Accuracy on the Training Dataset: 74.87%
- Model Accuracy on the Validation Dataset: 75.97%
- Model Accuracy on the Test Dataset: 63.28%

## Predicted Label Outcomes
Randomly predicted labels from the training set:
![img4](/docs/img4.png)
![img5](/docs/img5.png)