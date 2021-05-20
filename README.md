# FaceMaskDetector_Sparks

## Table of Contents: 
* Overview of Project

* Data Description 
* Libraries used

* Structure of the Approach

* Conclusion



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

* ### 1. Defining Transformations for train and test data
As different images have different sizes, it is important to define a tranform function which can take different images and return images which have similar aspect ratios and size. We will be using functions from <tt> torchvision.transforms </tt> like <tt>  transforms.Resize( )</tt> , <tt> transforms.CenterCrop() </tt> for this purpose. Also the pictures will be in form of pixels. We need to convert that to tensor values. For that we will use the function <tt> transforms.ToTensor() </tt><br>
Apart from these transformations which will be applicable for both train and test data ,  we will also apply certain transformations for loading train data like <tt> transforms.RandomRotation() </tt> and <tt> transforms.RandomHorizontalFlip() </tt> because the size of our training set is very smaall i.e 18743 , so these transformations will help to augment the size of data set 
 
* ### 2. Load the train and test data into notebook and applying the described transformations and divide them into minibatches

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
