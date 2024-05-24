# Unet_models

# Real-Time Image Segmentation with U-Net

This repository contains scripts for real-time image segmentation using a pre-trained U-Net model. The segmentation is performed on images captured from a webcam or selected from the filesystem, guiding directional decisions based on the position of segmented objects.

## Features
- Real-time image segmentation using a webcam.
- Image segmentation from selected image files.
- Directional guidance (left or right) based on object position.

## Files in the Repository
- `Camera based prediction.py`: Script for capturing real-time frames from a webcam, segmenting the images, and determining directional guidance based on object position.
- `Image Browse and predict.py`: Script for selecting an image file, performing segmentation, and displaying the results along with directional guidance.
- `multi_unet_model.py`: Script defining the multi-class U-Net model used for segmentation.
- `MixedField_V2.hdf5`: Pre-trained weights for the U-Net model.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Keras
- Matplotlib
- scikit-image

## Installation
1. Clone this repository:
    ```sh
    git clone https://github.com/TechDudeSid/Unet_models.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Unet_models
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Real-Time Segmentation from Webcam
1. Ensure your webcam is connected.
2. Run the `Camera based prediction.py` script:
    ```sh
    python Camera\ based\ prediction.py
    ```
3. The script will capture frames from the webcam, perform segmentation, and print "turn Left" or "turn Right" based on the position of the segmented object.

### Segmentation from Selected Image Files
1. Run the `Image Browse and predict.py` script:
    ```sh
    python Image\ Browse\ and\ predict.py
    ```
2. A file dialog will appear. Select an image file to process.
3. The script will display the original image, the segmentation result, and the directional guidance (left or right).

## Model Details
The U-Net model used in this project is a convolutional neural network designed for biomedical image segmentation. It uses pre-trained weights loaded from `MixedField_V2.hdf5`. The model architecture is defined in the `multi_unet_model.py` script, which builds a multi-class U-Net model with customizable input dimensions and number of classes.

### Example Results
Here are some example segmentation results along with the corresponding directional guidance:

**Results**
![Results](images/resuts.jpg)



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The U-Net model implementation is adapted from the [original paper](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
- This project utilizes the Keras deep learning library and OpenCV for image processing.
