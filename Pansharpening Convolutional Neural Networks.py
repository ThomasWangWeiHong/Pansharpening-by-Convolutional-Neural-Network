import cv2
import gc
import glob
import numpy as np
import rasterio
from keras.models import Input, Model
from keras.layers import Conv2D
from keras.optimizers import Adam



def training_image_creation(img_ms, img_pan, n_factor):
    """ 
    This function generates the blurred version of the original input multispectral image, and concatenate it with the 
    downsampled panchromatic so as to create the training sample used for Pansharpening Convolutional Neural Network (PCNN) 
    model training. 
    
    Inputs:
    - img_ms: Numpy array of the original multispectral image which is to be used for PCNN model training
    - img_pan: Numpy array of the original panchromatic image which is to be used for PCNN model training
    - n_factor: The ratio of pixel resolution of multispectral image to that of the panchromatic image
    
    Outputs:
    - training_sample_array: Numpy array of concatenated blurred multispectral image and downsampled panchromatic image to be 
                             used for PCNN model training
    
    """
    
    blurred_img_ms = np.zeros((img_ms.shape))
    
    for i in range(img_ms.shape[2]):
        blurred_img_ms[:, :, i] = cv2.GaussianBlur(img_ms[:, :, i], (5, 5), 0)
    
    blurred_img_ms_small = cv2.resize(blurred_img_ms, (int(img_ms.shape[1] / n_factor), int(img_ms.shape[0] / n_factor)), 
                                      interpolation = cv2.INTER_AREA)
    blurred_img_ms_sam = cv2.resize(blurred_img_ms_small, (img_ms.shape[1], img_ms.shape[0]), interpolation = cv2.INTER_CUBIC)
    
    downsampled_img_pan = cv2.resize(img_pan, (img_ms.shape[1], img_ms.shape[0]), 
                                     interpolation = cv2.INTER_AREA)[:, :, np.newaxis]
    
    training_sample_array = np.concatenate((blurred_img_ms_sam, downsampled_img_pan), axis = 2)
    
    return training_sample_array



def image_clip_to_segment(image_ms_array, train_image_array, image_height_size, image_width_size, percentage_overlap, 
                          buffer):
    """ 
    This function is used to cut up original input images of any size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire original input multispectral image and its corresponding 
    training image in the form of fixed size segments as training data inputs for the PCNN model.
    
    Inputs:
    - image_ms_array: Numpy array of original input multispectral image to be used for PCNN model training
    - train_image_array: Numpy array of training sample images to be used for PCNN model training
    - image_height_size: Height of image to be fed into the PCNN model for training
    - image_width_size: Width of image to be fed into the PCNN model for training
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by reflected values for positions with no valid data values
    
    Output:
    - train_segment_array: 4 - Dimensional numpy array of training sample images to serve as training data for PCNN model
    - image_ms_segment_array: 4 - Dimensional numpy array of original input multispectral image to serve as target data for 
                           training PCNN model
    
    """
    
    y_size = ((image_ms_array.shape[0] // image_height_size) + 1) * image_height_size
    y_pad = int(y_size - image_ms_array.shape[0])
    x_size = ((image_ms_array.shape[1] // image_width_size) + 1) * image_width_size
    x_pad = int(x_size - image_ms_array.shape[1])
    
    img_complete = np.pad(image_ms_array, ((0, y_pad), (0, x_pad), (0, 0)), mode = 'symmetric').astype(image_ms_array.dtype)
    train_complete = np.pad(train_image_array, ((0, y_pad), (0, x_pad), (0, 0)), 
                            mode = 'symmetric').astype(train_image_array.dtype)
        
    img_list = []
    train_list = []
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_ms_array.shape[2]]
            img_list.append(img_original)
            train_original = train_complete[i : i + image_height_size, j : j + image_width_size, :]
            train_list.append(train_original)
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_ms_array.shape[2]))
    train_segment_array = np.zeros((len(train_list), image_height_size, image_width_size, train_image_array.shape[2]))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        train_segment_array[index] = train_list[index]
        
    return train_segment_array, image_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to read in files from a folder which contains the images which are to be used for training the 
    PCNN model, then returns 2 numpy arrays containing the training and target data for all the images in the folder so that
    they can be used for PCNN model training.
    
    Inputs:
    - DATA_DIR: File path of the folder containing the images to be used as training data for PCNN model.
    - img_height_size: Height of image segment to be used for PCNN model training
    - img_width_size: Width of image segment to be used for PCNN model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by reflected values for positions with no valid data values
    
    Outputs:
    - train_full_array: 4 - Dimensional numpy array of concatenated multispectral and downsampled panchromatic images to serve as 
                            training data for PCNN model
    - img_full_array: 4 - Dimensional numpy array of original input multispectral image to serve as target data for training PCNN model
    
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_MS_files = glob.glob(DATA_DIR + '\\Train_MS' + '\\Train_*.tif')
    img_PAN_files = glob.glob(DATA_DIR + '\\Train_PAN' + '\\Train_*.tif')
    
    img_array_list = []
    train_array_list = []
    
    for file in range(len(img_MS_files)):
        
        with rasterio.open(img_MS_files[file]) as f:
            metadata = f.profile
            ms_img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        with rasterio.open(img_PAN_files[file]) as g:
            metadata_pan = g.profile
            pan_img = g.read(1)
            
        ms_to_pan_ratio = metadata['transform'][0] / metadata_pan['transform'][0]
            
        train_img = training_image_creation(ms_img, pan_img, n_factor = ms_to_pan_ratio)
    
        train_array, img_array = image_clip_to_segment(ms_img, train_img, img_height_size, img_width_size, 
                                                       percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        train_array_list.append(train_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    train_full_array = np.concatenate(train_array_list, axis = 0)
    
    del img_MS_files, img_PAN_files
    gc.collect()
    
    return train_full_array, img_full_array



def pcnn_model(image_height_size, image_width_size, n_bands, n1 = 64, n2 = 32, f1 = 9, f2 = 5, f3 = 5, l_r = 0.0001):
    """ 
    This function creates the PCNN model which needs to be trained, following the main architecture as described in the 
    paper 'Pansharpening by Convolutional Neural Networks' by Masi G., Cozzolino D., Verdoliva L., Scarpa G. (2016)
    
    Inputs:
    - image_height_size: Height of image segment to be used for PCNN model training
    - image_width_size: Width of image segment to be used for PCNN model training
    - n_bands: Number of channels contained in the input images (multispectral bands and panchromatic band)
    - n1: Number of filters for the first hidden convolutional layer
    - n2: Number of filters for the second hidden convolutional layer
    - f1: size of kernel to be used for the first convolutional layer
    - f2: size of kernel to be used for the second convolutional layer
    - f3: size of kernel to be used for the last convolutional filter
    - l_r: Learning rate to be used by the Adam optimizer

    Outputs:
    - model: PCNN model compiled using the parameters defined in the input, and compiled with the Adam optimizer and 
             mean squared error loss function
    
    """
    
    img_input = Input(shape = (image_height_size, image_width_size, n_bands))
    conv1 = Conv2D(n1, (f1, f1), padding = 'same', activation = 'relu')(img_input)
    conv2 = Conv2D(n2, (f2, f2), padding = 'same', activation = 'relu')(conv1)
    conv3 = Conv2D(n_bands - 1, (f3, f3), padding = 'same')(conv2)
    
    model = Model(inputs = img_input, outputs = conv3)
    model.compile(optimizer = Adam(lr = l_r), loss = 'mse', metrics = ['mse'])
    
    return model



def image_model_predict(input_ms_image_filename, input_pan_image_filename, output_filename, 
                        img_height_size, img_width_size, fitted_model, 
                        percentage_overlap, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for upsampling. The 
    output upsampled segment is then allocated to its corresponding location in the image in order to obtain the complete upsampled 
    image, after which it can be written to file.
    
    Inputs:
    - input_ms_image_filename: File path of the multispectral image to be pansharpened by the PCNN model
    - input_pan_image_filename: File path of the panchromatic image to be used by the PCNN model
    - output_filename: File path to write the file
    - img_height_size: Height of image segment to be used for PCNN model pansharpening
    - img_width_size: Width of image segment to be used for PCNN model pansharpening
    - ms_to_pan_ratio: The ratio of pixel resolution of multispectral image to that of panchromatic image
    - fitted_model: Keras model containing the trained PCNN model along with its trained weights
    - percentage_overlap: Percentage of overlap between adjacent patches of image for model prediction
    - write: Boolean indicating whether to write the pansharpened image to file
    
    
    Output:
    - pred_img_final: Numpy array which represents the pansharpened image
    
    """
    
    with rasterio.open(input_ms_image_filename) as f:
        metadata = f.profile
        ms_img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
    
    with rasterio.open(input_pan_image_filename) as g:
        metadata_pan = g.profile
        pan_img = np.expand_dims(g.read(1), axis = 2)
    
    ms_to_pan_ratio = metadata['transform'][0] / metadata_pan['transform'][0]
    ms_img_upsampled = cv2.resize(ms_img, (int(ms_img.shape[1] * ms_to_pan_ratio), int(ms_img.shape[0] * ms_to_pan_ratio)), 
                                  interpolation = cv2.INTER_CUBIC)
    pred_stack = np.concatenate((ms_img_upsampled, pan_img), axis = 2)
    
    
    y_size = ((pred_stack.shape[0] // img_height_size) + 1) * img_height_size
    y_pad = int(y_size - pred_stack.shape[0])
    x_size = ((pred_stack.shape[1] // img_width_size) + 1) * img_width_size
    x_pad = int(x_size - pred_stack.shape[1])
    
    img_complete = np.pad(pred_stack, ((0, y_pad), (0, x_pad), (0, 0)), mode = 'symmetric').astype(pred_stack.dtype)
    
    pred_img = np.zeros((img_complete.shape[0], img_complete.shape[1], ms_img.shape[2]))
    weight_mask = np.zeros((img_complete.shape[0], img_complete.shape[1], 1))
    img_holder = np.zeros((1, img_height_size, img_width_size, img_complete.shape[2]))

    
    for i in range(0, img_complete.shape[0] - img_height_size + 1, int((1 - percentage_overlap) * img_height_size)):
        for j in range(0, img_complete.shape[1] - img_width_size + 1, int((1 - percentage_overlap) * img_width_size)):
            img_holder[0] = img_complete[i : (i + img_height_size), j : (j + img_width_size), 0 : pred_stack.shape[2]]
            preds = fitted_model.predict(img_holder)
            pred_img[i : i + img_height_size, j : j + img_width_size, :] += preds[0, :, :, :]
            weight_mask[i : i + img_height_size, j : j + img_width_size, 0] += 1

    pred_img_complete = pred_img[0 : pan_img.shape[0], 0 : pan_img.shape[1], :]
    weight_mask_complete = weight_mask[0 : pan_img.shape[0], 0 : pan_img.shape[1], 0][:, :, np.newaxis]
    pred_img_final = (pred_img_complete / weight_mask_complete).astype(metadata['dtype'])

    
    metadata_pan['count'] = ms_img_upsampled.shape[2]
    with rasterio.open(output_filename, 'w', **metadata_pan) as dst:
        dst.write(np.transpose(pred_img_final, [2, 0, 1]))
    
    return pred_img_final
