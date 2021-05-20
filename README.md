# FaceMaskDetector_Sparks

## Table of Contents: 
* Overview of Project

* Data Description 
* Libraries used

* Structure of the Approach

* How to implement on your computer 



## Overview of Project:

We need to implement a face mask detector which detects whether a person is wearing a mask or not. This will be done for a video and an image

## Data Description:   
For this project , I will be using the data from a git hub repository by Prajna Bhandary  <a href='https://github.com/prajnasb/observations'>images_dataset</a>.<br>

Total number of images in dataset : 1376<br>
Number of images in train set : 690<br>
Number of images in test set : 686




## Libraries used:
* Numpy
* Matplotlib

* Tensorflow
* Keras<br>
  
* OpenCV

* os module of python
* sklearn
* imutils

## Structure of the Approach

* ### 1. mask_detector_training.ipynb
Firstly , we load the data set by accessing directories using the os module. Then after loading the image , we convert to to array using <tt>img_to_array</tt>. We then perform one hot encoding then split it into train and test sets. Now since our train set is very small, we augment the size of train set by applying transformations like rotation, shifting etc. Then we perform transfer learning by loading a MobileNetV2 network and make changes ensuring the head FC layer sets are left off. We then train the model , save it and see the performace
 
* ### 2. detecting_mask_in_image.ipynb

We load our pre trained model, then we load the image in which we need to detect, then we apply certain transformations to the input image and then load it into the model. On running the cells, we get an image with detection of wearing or not wearing the mask

* ### 3. mask_in_video.ipynb
We define a function that extracts blobs from the face. Then we perform resizing to the frame store 3 values faces,locs ,preds. Then we load our model and using imutils have a video stream wherein our previously defined function extracts every frame and predicts where the face is and then tell whether a person is wearing a mask or not.



## How to implement on your computer

Download the zip file of the repo. Then if you want to implement mask detector in image, open the detecting_mask_in_image.ipynb file, change the paths in the 2nd cell according to the path where you have downloaded the files . The path should correspond to the prototxt and resnet model file. Then in the 4th cell, change the "withmask.jpg" to the image file on which you want to perform the operation. Then you are good to go, just run all the cells


For video, just change the path in the 3rd cell corresponding the path where you store the prototext and resnet model file. After that just run the cells
