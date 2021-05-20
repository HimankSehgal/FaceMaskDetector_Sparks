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

Defining <tt> train_data </tt> and <tt> test_data </tt> and loading images from the path where they were saved in the computer along with applying the transformations described above. Also, since it is a common practice to divide the data into mini batches , so we will be doing the same using the <tt> torch.utils.data.DataLoader()</tt>

* ### 3. Defining the model class

The Model will be as follows 
We are having 3 types of layers with following specifications
1. Conv2d Layer : <tt> kernel_size = 4 x4 </tt>  , <tt> stride = 2</tt> ,<tt> padding = 0</tt> ( the output and input channels are mentioned below)
2. MaxPool2d Layer :<tt> kernel_size = 2x2 </tt>  , <tt> stride = 2</tt>
3. Full Connected layers : ( the output and input mentioned are mentioned below)

The flow of Model will be:-

Input ( 3 x 224 x 224) ---> (Conv2d Layer , MaxPool2d Layer) ---> 4 x 55 x 55 ---> (Conv2d Layer , MaxPool2d Layer) ---> 8 X 13 X 13 --->  ( Conv2d layer) ---> 16 x 5 x 5 ---> 
400 x 1 ---> FC layer(in_features = 400 ,out_features = 120) ---> FC layer(in_features = 120 ,out_features = 84) ---> FC layer(in_features = 84 ,out_features = 2), softmax_layer ---> predictions


I have kept number of channels and parameters like kernel_size , padding etc. in the powers of 2 as these help to speed up things because of structure of computer memory
The paramaters for fully connected layers are kept to be 120,84 as many architectures like LeNet-5 have followed this pattern

* ### 4. Instantiating the Model and defining the criteria for loss and optimizer 

We are using the criteria as <tt> nn.CrossEntropyLoss() </tt> and optimizer as <tt> torch.optim.Adam() </tt> with learning rate as 0.01

* ### 5. Performing Forward Propagation

Performed 20 epochs and stored values of train_loss, test_loss, train_accuracy ,test_accuracy for every epoch for plotting graphs later on

* ### 6. Evaluating performance
Plotted graphs to see the pattern of different parameters that were stored in a list during forward propagation

## Conclusion

The training loss was 0.58 and test loss was 0.30 in the end
