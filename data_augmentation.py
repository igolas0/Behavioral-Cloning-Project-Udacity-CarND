import math
import pandas
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
            image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image

def preprocess_image_file_train(line_data):
    i_clr = np.random.randint(3)
    correction = 0.25
    if (i_clr == 0):
        shift_ang = 0.
    if (i_clr == 1):
        shift_ang = correction
    if (i_clr == 2):
        shift_ang = -correction
    y_steer = float(line_data[3]) + shift_ang
    source_path = line_data[i_clr]
    #source_path = line_data[0]
    filename = source_path.split('/')[-1]
    #current_path = '/home/igolaso/Desktop/data/6lapstrack1/IMG/' + filename
    current_path = '/home/igolaso/Desktop/data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = augment_brightness_camera_images(image)
    image = add_random_shadow(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    
    return image,y_steer

def generator(samples,batch_size = 256):
    
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            i_line = np.random.randint(num_samples)
            line_data = samples[i_line]
            
            images = []
            measurements = []
            keep_pr = 0

            while keep_pr == 0:
                x,y = preprocess_image_file_train(line_data)
                pr_unif = np.random
                if abs(y)<.1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1
            
            images.append(x)
            measurements.append(y)

        X_train = np.array(images)
        y_train = np.array(measurements)
        yield X_train, y_train

samples = []
#with open('/home/igolaso/Desktop/data/6lapstrack1/driving_log.csv') as csvfile:
with open('/home/igolaso/Desktop/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)


#split samples into training and validation sets (80% to 20%)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


#shuffle samples
train_samples = sklearn.utils.shuffle(train_samples)
validation_samples = sklearn.utils.shuffle(validation_samples)

train_generator = generator(train_samples, batch_size = 16384)
validation_generator = generator(validation_samples, batch_size = 64)

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#from .losses import mean_squared_error
#from .losses import mean_absolute_error

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,15),(0,0))))
model.add(Convolution2D(16,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(32,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(128,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(256,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(8096))
#model.add(Dropout(0.5))
model.add(Dense(800))
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(1))

# Aliases

model.compile(loss='mse', optimizer='adam')

#model = load_model('model.h5')

pr_threshold = 0.5

model.fit_generator(train_generator, samples_per_epoch = 16384,
                    validation_data=validation_generator, 
                   nb_val_samples= len(validation_samples), nb_epoch=5)

model.save('model.h5')
