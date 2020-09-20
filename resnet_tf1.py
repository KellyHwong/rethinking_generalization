#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-09-20 19:21
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org


import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import slim
# tensorflow models repo
from models.research.object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1
from models.research.slim.nets import nets_factory

N_CLASSES = 10
DATA_SET_SIZE = 10000  # 50000
RESNET_VERSIONS = ['resnet_v1_200', 'resnet_v1_101', 'resnet_v1_50']

# Optional models
_ = ['resnet_v1_200', 'resnet_v1_152', 'resnet_v1_101', 'resnet_v1_50', 'resnet_v2_50',
     'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200', 'vgg_a', 'vgg_16', 'vgg_19']


class MLP(object):
    def __init__(self, resnet_version, random_labels, block_k):
        '''
        block_k : blocks at k and above will be unfrozen in the resnet. If you don't want to freeze any weights thsi should be zeo
        '''
        cwd = os.getcwd()
        self.resnet_version = resnet_version
        self.summaries_dir = cwd + '/summaries/' + self.resnet_version
        self.summaries_dir = self.summaries_dir + '/random_labels' if random_labels \
            else self.summaries_dir + '/normal'
        self.model_dir = cwd + '/models/' + self.resnet_version
        self.model_dir = self.model_dir + '/random_labels/mnist_ckpt_block{}'.format(block_k) \
            if random_labels else self.model_dir + '/normal/mnist_ckpt_block{}'.format(block_k)
        self.imgs = tf.placeholder(tf.float32, [None, None, None, 3])
        self.feature_extractor()
        self.build()

        self.labels = tf.placeholder(tf.int64, [None])
        y = tf.one_hot(self.labels, depth=N_CLASSES, dtype=tf.int64)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits))
        opt = tf.train.AdamOptimizer()  # PLAY W/ LEARNING RATE HERE!
        self.grads = opt.compute_gradients(self.loss)
        self.grads = [(g, v) for (g, v) in self.grads if self.grad_filter(
            block_k=block_k, v=v)]  # if 0 update everything but logits
        self.update = opt.apply_gradients(self.grads)

        n_equals = tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(
            self.logits, axis=1)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(n_equals)

        self.sess = tf.Session()
        self.summaries()
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def grad_filter(self, block_k, v):
        '''
        blocks some gradients from being applied(useful for block freezing)
        '''
        if block_k == 0:
            if 'logits' not in v.name:
                return True
        if 'classifier' in v.name:
            return True
        for k in range(block_k, 5, 1):  # max block number is 4
            if 'block{}'.format(k) in v.name:
                return True
        return False

    def summaries(self):
        '''
        used for tensorboard summaries
        '''
        # tf summaries
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('cross_entropy_loss', self.loss)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            self.summaries_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.summaries_dir + '/test')

    def feature_extractor(self):
        '''
        this function generates features from base residual network. We piggyback off of
        tensorflows resnet implementation. num_classes is 1k b/c it easily allows us to plug in 
        pretrained weights which were learned on a dataet w/ 1k classes. However, we only take
        output from block4

        PLAY WITH RESNET SIZE HERE! check tf resnet code to see how to easily chaange resnet size.
        currently it is 50
        '''
        network_fn = nets_factory.get_network_fn(self.resnet_version,
                                                 num_classes=1000, is_training=False)  # 1k b/c of pretrained weight shape
        _, endpoints = network_fn(self.imgs)
        self.features = tf.reshape(
            endpoints[self.resnet_version+'/block4'], [tf.shape(self.imgs)[0], 2048])
        self.res_variables_to_restore = slim.get_variables_to_restore()

    def build(self):
        '''
        this is  a head we drop on top of the base resnet. This is the part that actually does
        classification.
        '''
        with tf.variable_scope('classifier'):
            self.l1 = slim.fully_connected(
                self.features, 200, activation_fn=tf.nn.relu)
            self.l2 = slim.fully_connected(
                self.l1, 200, activation_fn=tf.nn.relu)
            self.logits = tf.layers.dense(self.l2, N_CLASSES, activation=None)

    def preprocess_imgs(self, imgs):
        '''
        format the images
        parameters
        ----------
        imgs : list of images loaded from opencv2

        returns
        -------
        transformed_imgs : images that are reformatted for training
        '''
        transformed_imgs = []
        for i in range(len(imgs)):
            # [batch, height, width, channels]
            new_img = np.reshape(imgs[i], [28, 28, 1])
            # copy channel b/c mnist is grayscale
            new_img = np.repeat(new_img, 3, axis=2)
            transformed_imgs.append(new_img)
        transformed_imgs = np.reshape(transformed_imgs, [-1, 28, 28, 3])
        return transformed_imgs

    def test(self, labels, imgs, i):
        imgs = self.preprocess_imgs(imgs)
        test_summary = self.sess.run(self.merged,
                                     feed_dict={self.imgs: imgs, self.labels: labels})
        self.test_writer.add_summary(test_summary, i)

    def train(self, labels, imgs, i):
        imgs = self.preprocess_imgs(imgs)
        if i % 10 == 0:
            y_hat, _, summary = self.sess.run([self.logits, self.update, self.merged],
                                              feed_dict={self.imgs: imgs, self.labels: labels})
            self.train_writer.add_summary(summary, i)
        else:
            y_hat, _ = self.sess.run([self.logits, self.update],
                                     feed_dict={self.imgs: imgs, self.labels: labels})

    def save(self, i):
        self.saver.save(self.sess, self.model_dir+'/mnist', global_step=i)

    def load(self):
        ckpt_path = tf.train.latest_checkpoint(self.model_dir)
        if ckpt_path is not None:
            self.saver.restore(self.sess, ckpt_path)
        return 0 if ckpt_path is None else int(ckpt_path.split('-')[-1])


def train(resnet_version, random_labels, freeze_before_k):
    '''
    parameters
    ----------
    random_labels : if random labels is true we randomly assign labels to the training set
    freeze_before_k : all blocks before k will not apply gradients(k=0 will train as normal)
    '''
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    mnist = input_data.read_data_sets(
        'fashion-mnist/data/fashion')  # load fashion mnist

    mlp = MLP(resnet_version, random_labels, block_k=freeze_before_k)
    i = mlp.load()
    batch_size = 128
    n_epochs = 10
    if random_labels is True:
        mnist.train._labels = np.random.randint(
            0, N_CLASSES, mnist.train.labels.shape[0])

    # shrink  the data set for easier memorization
    # use a subset of training set so memorizing is possible
    data_set_size = DATA_SET_SIZE
    n_epochs = n_epochs * mnist.train._labels.shape[0] / data_set_size
    mnist.train._labels = mnist.train._labels[:data_set_size]
    mnist.train._images = mnist.train._images[:data_set_size]
    mnist.train._num_examples = data_set_size

    while(mnist.train._epochs_completed < n_epochs and i*batch_size < (n_epochs * 50000)):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        mlp.train(batch_y, batch_x, i)
        i += 1
        if i % 100 == 0:
            batch_test_x, batch_test_y = mnist.test.next_batch(100)
            mlp.test(batch_test_y, batch_test_x, i)
            ''' 
            mlp.save(i)
            print('saving after {} iterations'.format(i))
            '''
            print('EPOCH: ', mnist.train._epochs_completed)

    return mlp

# test effects on all 3 networks


resnet_versions = RESNET_VERSIONS
for resnet in resnet_versions:
    tf.reset_default_graph()
    train(resnet, random_labels=True, freeze_before_k=0)
    tf.reset_default_graph()
    train(resnet, random_labels=False, freeze_before_k=0)
