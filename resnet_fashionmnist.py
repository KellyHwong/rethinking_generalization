#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-06-20 12:31
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras  # keras-tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.utils import plot_model


from resnet import model_depth, resnet_v2, lr_schedule
from utils.dir_utils import makedir_exist_ok


def main():
    # data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), \
        (test_images, test_labels) = fashion_mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    padded = True
    if padded:
        train_images = np.load("./mnist_train_images_padded.npy")
    train_labels = np.eye(num_classes)[train_labels]

    # model
    selected_model = "ResNet20v2"
    # selected_model = "keras.applications.ResNet50V2"
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)

    metrics = [Accuracy()]

    train_images = np.expand_dims(train_images, -1)
    input_shape = train_images.shape[1:]

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
    else:
        return

    # plot model
    plot_model(model, to_file=f"{selected_model}.png", show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),  # learning_rate=lr_schedule(0)
                  metrics=metrics)

    # checkpoint = ModelCheckpoint(filepath=filepath, monitor="acc",verbose=1)
    logdir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    csv_logger = CSVLogger(os.path.join(
        logdir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        logdir, histogram_freq=1)
    callbacks = [csv_logger, tensorboard_callback]
    # makedir_exist_ok()

    # fit
    epochs = 100
    batch_size = 32
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
