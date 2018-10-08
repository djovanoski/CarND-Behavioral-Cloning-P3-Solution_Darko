import cv2
import numpy as np
import pandas as pd
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from os.path import join
import os
from model import model
from image import ImageGenerator

#Keras
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler





DATA_CSV = '/home/workspace/CarND-Behavioral-Cloning-P3/data_org/data/driving_log.csv'
LEARNING_RATE =1e-3
DECAY = .5
PATIENCE = 5
BATCH_SIZE = 8
MODEL_NAME = 'model'

def load_train_test_data(csvpath=glob.glob(os.path.join('data', '*.csv'))):
    if getattr(csvpath, 'lower'):
        df = pd.read_csv(csvpath)
    elif len(csvpath) == 1:
        df = pd.read_csv(csvpath[0])
    elif len(csvpath) > 1:
        df = pd.concat(
            [pd.read_csv(f) for f in csvpath],
            ignore_index=True)
    else:
        raise ValueError('No csv files found')

    center = df.center.values
    speed = df.speed.values / df.speed.max()
    throttle = df.throttle.values
    y = df.steering.values
    c1, c2, s1,s2,t1, t2, y1, y2 = train_test_split(center, speed, throttle, y, test_size=0.2)
    print('chekcc',y2)
    return [c1, s1, t1, y1], [c2, s2, t2, y2]
def train():
    # Load dataset
    training_data,validation_data = load_train_test_data(DATA_CSV)
    #print(validation_data[3].shape)
    train_datagen = ImageGenerator(horizontal_flip=False, vertical_flip=True, rotation_range=0, rescale=127.5)
    valid_datagen = ImageGenerator(horizontal_flip=False, vertical_flip=False, rotation_range=0, rescale=127.5)

    # Keras inputs
    image = Input(shape=(160, 320, 3), name='image')
    velocity = Input(shape=(1,), name='velocity')

    # Keras model
    model = model(image, velocity)
    model.compile(optimizer=Adam(LEARNING_RATE), loss='mse', loss_weights=[1., .01])
    print(model.summary())

    # Training
    callbacks = [
        # Save best model
        ModelCheckpoint('./{}.h5'.format(MODEL_NAME), monitor='val_loss', save_best_only=True),
        # Stop training after 5 epochs without improvement
        EarlyStopping(monitor='val_loss', patience=PATIENCE),
        # Polynomial decaying learning rate
        LearningRateScheduler(lambda x: LEARNING_RATE * DECAY ** x)
    ]

    history = model.fit_generator(
        generator=train_datagen.flow_from_list(training_data,target_size=(160,320,3),batch_size=BATCH_SIZE, shuffle=True, seed=123),
        steps_per_epoch=training_data[3].size // BATCH_SIZE,
        epochs=100,
        callbacks=callbacks,
        validation_data=valid_datagen.flow_from_list(validation_data,target_size=(160,320,3),
                                                     batch_size=BATCH_SIZE, shuffle=False, seed=123),
        validation_steps=validation_data[3].size // BATCH_SIZE
    )

    #with open('{}.p'.format(MODEL_NAME), 'wb') as f:
    #    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train()
