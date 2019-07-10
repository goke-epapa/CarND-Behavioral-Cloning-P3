import csv
import cv2
from scipy import ndimage
import numpy as np

# Load training data from CSV file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for line in reader:
        if count != 0:
            samples.append(line)
        count += 1


def load_raw_data(csv_data):
    imgs = []
    measures = []

    for row in csv_data:
        source_path = row[0]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename

        image = ndimage.imread(current_path)
        imgs.append(image)
        measurement = float(row[3])
        measures.append(measurement)
        
    return imgs, measures


def augment_images(raw_images, raw_measurements):
    augmented_imgs, augmented_measures = [], []
    for image, measurement in zip(raw_images, raw_measurements):
        augmented_imgs.append(image)
        augmented_measures.append(measurement)
        augmented_imgs.append(cv2.flip(image, 1))
        augmented_measures.append(measurement * -1.0)
    return augmented_imgs, augmented_measures


# Load raw images
images, measurements = load_raw_data(samples)


# Augment images by flipping images and multiplying measurements by -1
augmented_images, augmented_measurements = augment_images(images, measurements)
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# Nvidia Self Driving Car CNN
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D

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

# Train model
model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# Save model to file
model.save('model.h5')
