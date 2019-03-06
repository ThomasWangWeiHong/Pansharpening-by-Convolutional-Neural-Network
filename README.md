# Pansharpening-by-Convolutional-Neural-Network
Python implementation of Convolutional Neural Network (CNN) proposed in academic paper

This repository contains functions to preprocess the separate training multispectral and panchromatic images so that they can be used
to train the CNN model. In addition, this repository also contains functions to generate a pansharpened image which is produced by the 
CNN using input multispectral and panchromatic images (of any size, and with the same extents) as inputs. 

The CNN used here is the Pansharpening Convolutional Neural Network (PCNN) implemented in the paper 
'Pansharpening by Convolutional Neural Networks' by Masi G., Cozzolino D., Verdoliva L., Scarpa G. (2016)

The main difference in the implementation in this repository and the one proposed in the paper is the use of the Adam optimizer.

Requirements:
- cv2
- gc
- glob
- numpy
- rasterio
- keras (TensorFlow backend)
