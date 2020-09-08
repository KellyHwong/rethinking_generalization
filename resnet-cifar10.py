#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-03-20 00:05
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://keras.io/zh/examples/cifar10_cnn/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from resnet import model_depth, resnet_v1, resnet_v2, lr_schedule


def main():
    # TODO record the learning rate vs epoch
    cifar10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_labels = keras.utils.to_categorical(train_labels)

    input_shape = train_images.shape[1:]
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20

    # model type
    selected_model = "ResNet20v2"
    # selected_model = "keras.applications.ResNet50V2"
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])

    # callbacks
    logdir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    csv_logger = CSVLogger(os.path.join(
        logdir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        logdir, histogram_freq=1)
    callbacks = [csv_logger, tensorboard_callback]

    model.fit(train_images,
              train_labels,
              epochs=epochs,
              validation_data=(test_images, test_labels),
              batch_size=batch_size,
              verbose=1, workers=4,
              callbacks=callbacks)


if __name__ == "__main__":
    main()
