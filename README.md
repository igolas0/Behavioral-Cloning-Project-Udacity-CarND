# Deep Learning to Clone Driving Behavior


[//]: # (Image References)

[image1]: ./output_images/test.png "Data Augmentation"
[image2]: ./output_images/test2.png "Data Augmentation"
[image3]: ./output_images/data.png "Data Distribution"

Overview
---

In this project a convolutional neural network is used to clone driving behavior. First driving data is collected using a simulator and then a convolutional neural network is trained and tested using Keras. The model will output a steering angle to an autonomous vehicle. Finally, I also do some experimentation with a model that outputs both steering angle and throttle values.

 The trained model drives the car autonomously around the track1 and the harder track2 of the simulator. Videos of the recorded laps driving autonomously are provided.


---
## Project

#### 1. Project Files Overview 

My project includes the following files:
* data_augmentation.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* drive_with_throttle.py for driving the car in autonomous mode using the trained model that also outputs throttle values.
* model.h5 containing a trained convolution neural network 
* model_throttle.h5 containing a trained convolution neural network that outputs steering angles and throttle values.
* README.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

"data_augmentation.py" contains the CNN model. The file shows the pipeline I used for training and validating the model.

#### 2. Model Architecture and Training Strategy

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 256 (model.py lines 129-148) 

The model includes RELU layers to introduce nonlinearity (code line 132,134,136,138,140), and the data is normalized in the model using a Keras lambda layer (code line 130). 

On top of the convolutional layers I placed 3 fully connected layers with 8096, 800 and 100 neurons respectively before the final 1 neuron output (which predicts the steering angle based on the input image).

#### 3. Managing Overfitting

While I experimented with some dropout layers I did not use them, since overfitting was not a problem as long as I did not train the model for too much epochs (5 epochs - each epoch with 16384 training generated samples was enough).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 110-119 and 152-162).

 The model was tested by running it after every epoch through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (data_augmentation.py line 152).

#### 4. Training data

4 laps were recorded with center lane driving and no recoveries. Of the 4 laps 2 were recorded on each track (one reverse lap on each track).

 I tried to train a model on only track 1 and with help of data augmentation generalize well on to track 2. For now I have not been able to achieve this with only track 1 data and will require further experimentation and tweaking (also other techniques for data augmentation). Hence for this project I used both track 1 and track 2 training data.

My first step was to use a convolution neural network model similar to the NVIDIA paper ["End to End Learning for Self-Driving-Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The main differences are that I use 3x3 kernels only while in NVIDIA's paper 5x5 kernels are used the after the first 3 layers. Then I also extract more depth out of my convolutional layers and go up to 256 convolutional filters.

The reason of using 3x3 filters is that in several Imagenet competition papers it is generally believed to produce better accuracy. With a higher filter depth and more neurons on the fully connected layers we have a higher capacity of extracting information. We might need this since our input images have more pixels than NVIDIA's (80x320 after cropping instead of 66x200), but probably we would be also good using the same architecture as NVIDIA's.

The sampled data was split into 80% training images and 20% test/validation set and then shuffled.

#### 5. Data Augmentation

To augment the data I used several techniques.

First I used the three cameras on the top of the car (center, left and right). A correction steering angle for left and right images was used (+0.2 and -0.2 respectively) so that the car learns to correct its trajectory towards the center of the lane. This happens in lines #47 to #61 of the code (notice that center/left/right cameras each has 33% probability of being used).

Then I also introduced a 50% of probability for the images to getting flipped (while multiplying the steering angle by factor -1 for flipped images). Lines of code #66 to #69.

Finally I also used random shadow and brightness shifts. I got this idea from this awesome [post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) by Vivek.

Here are a few examples of the images generated with this approach:

![alt text][image2]

For me the big takeaway is that data augmentation does not only provide a way around small datasets but maybe even more importantly it helps the neural network generalize much better to unseen data by restricting the neural net from making "foolish assumptions". Here is a [good read](https://neil.fraser.name/writing/tank/), which makes this apparent. 

#### 6. Generator

The data generator which feeds the convolutional neural network during training is defined from lines #73 to #100 of the code. A probability threshold is used to initialize the training of the network with large steering angles (larger as absolute 0.1) and hence avoid a bias towards straight (zero steering angle) driving.

I used a threshold of 1.0 for the first epoch (which means 100% probability of not keeping steering angles smaller than abs(0.1) - before correction for left/right camera). Then I used a 0.4 threshold and trained the net for four more epochs.

Below the distribution of 2000 random selected samples by the generator depicted in a binned histogram. The x-axis corresponds to the steering angles:

![alt text][image3]


#### 7. Testing the model 

The final step was to run the simulator to see how well the car was driving around track one. After a lot of testing and iterations I was able to come with the model described above and successfully driving around track1 and 2.

Here is a video of the model driving autonomously around [Track1](./track1.mp4) and [Track2](./track2.mp4).

I also trained a model which outputs also the throttle value as well and successfully drives around [Track1 with throttle](./track1_throttle.mp4). On track 2 it does not come very far but it is a lot of fun to watch [Track2 with throttle](./track2_throttle.mp4). A big problem here comes from using the keyboard for speeding up (which produces unevenly distributed data) and also the signal itself which combines throttle (positive values) with the brake (negative values). Maybe some preprocessing of the signal and maybe even separating the throttle and brake functions into different signals could help here.


This project was lots of fun. I could spend forever optimizing this. Still I felt that it was difficult judging the robustness of the pipeline before testing on the simulator.

The main thing I would spend time on for further work on this project would be the data augmentation. I might skip using left and right cameras and instead use random translational shifts of the images (to the sides and also vertically to simulate road slope) while also correcting the steering angle depending of how big the shift was. It would be awesome to achieve generalizing to track 2 while training only on track 1 images.

## Details About Files In This Directory

The following resources can be found in this github repository:
* drive.py
* video.py
* README.md

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

