#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-01-20 16:52
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import datetime
import os
import sys
from optparse import OptionParser
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras  # keras-tf

from utils.dir_utils import makedir_exist_ok
from keras_fn.resnet import model_depth, resnet_v2, lr_schedule


def cmd_parser():
    parser = OptionParser()
    parser.add_option('--model_type', type='string', dest='model_type',
                      action='store', default="ResNet20v2", help='model_type, user named experiment name, e.g., ResNet20v2_BCE.')
    parser.add_option('--exper_type', type='string', dest='exper_type',
                      action='store', default="normal", help='Whether to randomized all train labels.')
    parser.add_option('--loss', type='string', dest='loss',
                      action='store', default="bce", help='loss name, e.g., bce or cce.')
    # Parameters we care
    parser.add_option('--batch_size', type='int', dest='batch_size',
                      action='store', default=32, help='batch_size, e.g. 16, 32.')
    parser.add_option('--epochs', type='int', dest='epochs',
                      action='store', default=200, help='training epochs, e.g. 150.')  # training 200 epochs to fit enough

    args, _ = parser.parse_args(sys.argv[1:])

    return args


def main():
    options = cmd_parser()

    # data, e.g., fashion_mnist = keras.datasets.fashion_mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), \
        (test_images, test_labels) = mnist.load_data()
    num_classes = np.max(train_labels) + 1  # 10

    # fit options
    epochs = options.epochs
    batch_size = options.batch_size

    # choose model
    # model_type = "keras.applications.ResNet50V2"
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    model_type = "ResNet%dv%d" % (depth, version)  # "ResNet20v2"

    metrics = [BinaryAccuracy(), CategoricalAccuracy()]

    padded = False
    if model_type == "ResNet20v2":
        padded = True  # True
    if padded:
        train_images = np.load("./mnist_train_images_padded.npy")

    train_images = np.expand_dims(train_images, -1)
    input_shape = train_images.shape[1:]

    exper_type = options.exper_type
    np.random.seed(42)
    if exper_type == "random_labels":
        print(f"exper_type is random_labels, randomized labels.")
        train_labels = np.random.randint(0, num_classes, train_labels.shape[0])
    elif exper_type == "shuffle_labels":
        print(f"exper_type is shuffle_labels, shuffle labels.")
        np.random.shuffle(train_labels)

    train_labels = keras.utils.to_categorical(train_labels)  # to one-hot

    if model_type == "ResNet20v2":
        model = resnet_v2(input_shape=input_shape,
                          depth=depth, num_classes=num_classes)
    elif model_type == "keras.applications.ResNet50V2":
        model = tf.keras.applications.ResNet50V2(
            include_top=True,
            weights=None,
            input_shape=input_shape,
            classes=num_classes
        )
    elif model_type == "FC":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # 28
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    else:
        return

    # plot model
    # plot_model(model, to_file=f"./fig/model/{model_type}.png", show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  #   metrics=['accuracy'])
                  metrics=metrics)  # metrics = [Accuracy()]

    # define callbacks
    # checkpoint = ModelCheckpoint(filepath=filepath, monitor="acc",verbose=1)
    logdir = os.path.join(
        "logs", model_type, exper_type, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(
        logdir + "/metrics")  # custom scalars
    file_writer.set_as_default()

    csv_logger = CSVLogger(os.path.join(
        logdir, "training.log.csv"), append=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        logdir, histogram_freq=1)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [csv_logger, tensorboard_callback, lr_scheduler]

    # fit model
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
