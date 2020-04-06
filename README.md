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


The following images illustrates the PCNN model in action on a sample GeoEye - 1 image (courtesy of European Space Agency), which is not one of the sensors included in the training dataset for the PCNN model. It serves to test the generalizability of the PCNN model on sensors on which the model has not been trained on before. Do note that the default parameters are used to train this particular PCNN model, and that some improvements might be expected if fine - tuning of the parameters are conducted.


Sample GeoEye - 1 Multispectral Image (Courtesy of European Space Agency):
![Alt Text](https://github.com/ThomasWangWeiHong/Pansharpening-by-Convolutional-Neural-Network/blob/master/Test_Enlarged.JPG)


Sample GeoEye - 1 Panchromatic Image (Courtesy of European Space Agency):
![Alt Text](https://github.com/ThomasWangWeiHong/Pansharpening-by-Convolutional-Neural-Network/blob/master/Test_Enlarged_Pan.JPG)


Sample GeoEye - 1 Pansharpened Image:
![Alt Text](https://github.com/ThomasWangWeiHong/Pansharpening-by-Convolutional-Neural-Network/blob/master/Test_Enlarged_PSH.JPG)
