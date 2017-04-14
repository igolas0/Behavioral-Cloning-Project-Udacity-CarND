import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

#load recorded data from "driving_log.csv"
samples = []
with open('/home/igolaso/Desktop/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

#shuffle samples
samples = sklearn.utils.shuffle(samples)

#split samples into training and validation sets (80% to 20%)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples)+ len(validation_samples))

#define generator to feed neural net input on demand (in batches) instead
# of storing data in memory
def generator(samples, batch_size=24):
        num_samples = len(samples)
        while 1: # Loop forever
           #sklearn.utils.shuffle(samples)
           for offset in range(0, num_samples, int(batch_size/6)):
               batch_samples = samples[offset:offset+batch_size]
               images = []
               measurements = []

               for batch_sample in batch_samples:
                   steering_center = float(batch_sample[3])
	           #create adjusted steering angles for side camera images
                   correction = 0.2
                   steering_left = steering_center + correction
                   steering_right = steering_center - correction
                   measurements.extend((steering_center, steering_center*-1.0, steering_left, steering_left*-1.0, steering_right, steering_right*-1.0))

                   for i in range(3):
                       source_path = batch_sample[i]
                       filename = source_path.split('/')[-1]
                       current_path = '/home/igolaso/Desktop/data/IMG/' + filename
                       image = cv2.imread(current_path)
                       image_flipped = np.fliplr(image)
                       images.extend((image, image_flipped))

               X_train = np.array(images)
               y_train = np.array(measurements)
               yield X_train, y_train


train_generator = generator(train_samples, batch_size = 24)
validation_generator = generator(validation_samples, batch_size = 24)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

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
model.add(Dense(120))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch = 6*len(train_samples),
                    validation_data=validation_generator, 
                   nb_val_samples= 6*len(validation_samples), nb_epoch=10)

model.save('model.h5')
