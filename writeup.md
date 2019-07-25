# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_image1.jpg "Center Image"
[image2]: ./images/center_image2.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing the autonomous driving of car

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 59-71) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 70). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 79). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 87). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and augmented data which flipped the center images and use the negative of the measurements 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage an existing architecture and improve it

My first step was to use a convolution neural network model similar to the [NVIDIA end to end deep learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) I thought this model might be appropriate because it was created for the purpose of self driving cars

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the over-fitting, I modified the model and added a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I improved the training by reducing the number of epochs to 5

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The final results of the model are shown below

```
25212/25212 [==============================] - 31s 1ms/step - loss: 0.0155 - val_loss: 0.0099
Epoch 2/5
25212/25212 [==============================] - 29s 1ms/step - loss: 0.0143 - val_loss: 0.0094
Epoch 3/5
25212/25212 [==============================] - 29s 1ms/step - loss: 0.0139 - val_loss: 0.0087
Epoch 4/5
25212/25212 [==============================] - 29s 1ms/step - loss: 0.0137 - val_loss: 0.0087
Epoch 5/5
25212/25212 [==============================] - 29s 1ms/step - loss: 0.0134 - val_loss: 0.0086
```

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-82) consisted of a convolution neural network with the following layers:
| Layer         		    |     Description	        				                | 
|:-------------------------:|:---------------------------------------------------------:| 
| Input         		    | 65x320x3 RGB image   					                    | 
| Convolution 5x5     	    | 2x2 stride, valid padding, depth 24, activation relu      |
| Convolution 5x5     	    | 2x2 stride, valid padding, depth 36, activation relu      |
| Convolution 5x5     	    | 2x2 stride, valid padding, depth 48, activation relu      |
| Convolution 5x5     	    | valid padding, depth 64, activation relu                  |
| Convolution 5x5     	    | valid padding, depth 64, activation relu                  |
| Flatten	      	        | outputs 576 				                                |
| Fully connected		    | input 100, output 50        					            |
| Dropout       		    | 0.5                       					            |
| Fully connected		    | input 50, output 10        					            |
| Fully connected (logits)	| input 10, output 1        				                |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded over two laps on track one using center lane driving, while slowing down at corners. Here is an example image of center lane driving:

![alt text][image1]

Then I repeated this process on track one this time, I did not slow down at corners.

I also added the sample training data provided to have more data points.

To augment the data set, I flipped images and angles. For example, here is an image that has then been flipped:

![alt text][image2]

After the collection and augmentation process, I had 25212 number of data points. I then preprocessed this data by normalising the images i.e. dividing by 255.0 and deducting 0.5, then I cropped 70 pixels from the top and 25 pixels from the bottom.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
