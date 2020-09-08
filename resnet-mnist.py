#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-01-20 16:52
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras  # keras-tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, LearningRateScheduler
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.utils import plot_model

from resnet import model_depth, resnet_v2, lr_schedule
from utils.dir_utils import makedir_exist_ok


def main():
    # data
    # fashion_mnist = keras.datasets.fashion_mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), \
        (test_images, test_labels) = mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    # fit options
    epochs = 200
    batch_size = 32

    # model
    selected_model = "ResNet20v2"
    # selected_model = "keras.applications.ResNet50V2"
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)

    metrics = [BinaryAccuracy(), CategoricalAccuracy()]

    padded = False
    if selected_model == "ResNet20v2":
        padded = True  # True
    if padded:
        train_images = np.load("./mnist_train_images_padded.npy")

    train_images = np.expand_dims(train_images, -1)
    input_shape = train_images.shape[1:]
    train_labels = keras.utils.to_categorical(train_labels)

    if selected_model == "ResNet20v2":
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
    elif selected_model == "keras.applications.ResNet50V2":
        model = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )
    elif selected_model == "FC":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # 28
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    else:
        return

    # plot model
    # plot_model(model, to_file=f"./fig/model/{selected_model}.png", show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  #   metrics=['accuracy'])
                  metrics=metrics)  # metrics = [Accuracy()]

    # define callbacks
    # checkpoint = ModelCheckpoint(filepath=filepath, monitor="acc",verbose=1)
    logdir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), selected_model)
    file_writer = tf.summary.create_file_writer(
        logdir + "/metrics")  # custom scalars
    file_writer.set_as_default()

    csv_logger = CSVLogger(os.path.join(
        logdir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        logdir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [csv_logger, tensorboard_callback, lr_scheduler]

    # fit
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
