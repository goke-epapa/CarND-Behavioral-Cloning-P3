import csv
import cv2
from scipy import ndimage
import numpy as np

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for line in reader:
        if count != 0:
            samples.append(line)
        count += 1

def load_raw_data(csv_data):
    images = []
    measurements = []

    for line in csv_data:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename

        image = ndimage.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        
    return images, measurements

# Augment images
def augment_images(raw_images, raw_measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(raw_images, raw_measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)
    return augmented_images, augmented_measurements

import sklearn
from sklearn.utils import shuffle

# Generator 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split('/')[-1]
                current_path = './data/IMG/' + filename

                image = ndimage.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                angles.append(measurement)
                
                # Augment images
                images.append(cv2.flip(image, 1))
                angles.append(measurement * -1.0)
            
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)
         
images, measurements = load_raw_data(samples)
augmented_images, augmented_measurements = augment_images(images, measurements)
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from sklearn.model_selection import train_test_split
def get_generators(batch_size=1000):
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return generator(train_samples, batch_size=batch_size), generator(validation_samples, batch_size=batch_size) 

# use generator function
# train_generator, validation_generator = get_generators()

# Nvidia Self Driving Car Model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from math import ceil

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
# model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=3, verbose=1)

model.save('model.h5')
