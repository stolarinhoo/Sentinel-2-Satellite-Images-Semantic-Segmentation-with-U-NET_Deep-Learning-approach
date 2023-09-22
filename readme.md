# Sentinel-2 Satellite Images Semantic Segmentation with U-NET

Welcome to this repository, which encompasses the entirety of the codebase used for training and evaluating a U-NET model specialized in the semantic segmentation of Sentinel-2 satellite imagery. The repository was created on the basis of writing my master's thesis, the aim of which was to study land cover changes in and around urban area. 

## Environment and Library Installation (conda)

To run this project, it's recommended to set up a dedicated conda environment. You can do so with the following commands:

```bash
conda create -n sentinel2_unet python=3.10
conda activate sentinel2_unet
```

Once the environment is activated, you can install the necessary libraries with:

```bash
conda install --file requirements.txt
```

## Base Satellite Image

Sentinel-2 is an Earth observation satellite providing high-resolution images. These images are particularly useful for climate change research, vegetation monitoring, and other environmental applications.

S2B_MSIL2A_20180607T095029_N0208_R079_T33UYS_20180607T130225


## Classes

The model is trained to identify 8 classes:

1. Water
2. Urban Areas
3. Forests
4. Cultivated Fields
5. Grasslands
6. Desert Areas
7. Mountainous Regions
8. Snow/Ice-covered Regions

## Data Preparation for Training

1. **Data Acquisition**: Acquire data from appropriate sources like the Copernicus Open Access Hub.
2. **Pre-processing**: Normalize images, remove noise, and apply other preprocessing techniques.
3. **Data Splitting**: Split the data into training, validation, and test sets.
4. **Data Augmentation**: Apply augmentation techniques like rotation, scaling, and shifting to increase the diversity of training data.

## Training Process and Model Outcomes

We use the U-NET architecture for segmentation. The training process includes:

- Initializing the network with random weights.
- Setting the loss function and optimizer.
- Training the network on training data.
- Validating the model on validation data.
- Evaluating the model on test data.

The model's outcomes are presented in the form of learning curves, confusion matrices, and other evaluation metrics.

## Predicted Label Outcomes

After training the model, you can visualize its segmentation capabilities on test images. The predicted label outcomes are showcased in images where different classes are highlighted with distinct colors, making it easy to discern areas identified by the model.

---

Thank you for using this project! If you have any questions or feedback, please reach out via GitHub.