# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import os.path
import time
import importlib
import itertools
import math,pdb
import tensorflow as tf
import resnet_50_

from tensorflow.python.ops import data_flow_ops

image_directory = '/home/czw/图片/Places Pulse/'
# train_filename = '/home/czw/文档/csv/safety/train_safety.csv'
# validation_filename = '/home/czw/文档/csv/safety/validate_safety.csv'
train_filename = '/home/czw/文档/csv/safety/test_train.csv'
validation_filename = '/home/czw/文档/csv/safety/test_validation.csv'

def read_pairs_path_label_test(image_directory, csv_filename):  # return two lists of pairs of path of image and labels
    pairs = []
    labels = []
    labels_sigle = []
    data = np.genfromtxt(csv_filename, delimiter=',', dtype=str)
    for i in range(len(data)):
        if data[i][2] == 'left':
            flag = 1
        elif data[i][2] == 'right':
            flag = -1
        else:
            continue
        x = data[i][0]
        y = data[i][1]
        x_path = os.path.join(image_directory, x + '.jpg')
        y_path = os.path.join(image_directory, y + '.jpg')
        pair = [x_path, y_path]
        pairs.append(pair)
        labels.append([flag,flag])
        labels_sigle.append(flag)
    return np.array(pairs), np.array(labels), np.array(labels_sigle)


def read_image_path(filenames):
    images = []
    for filename in tf.unstack(filenames, axis=1):
        print(filename)
        file_contents = tf.read_file(filename)
        print(file_contents)
        image = tf.image.decode_image(file_contents, channels=3)
        image.set_shape((224, 224, 3))
        images.append(image)
    return np.array(images)

def log_loss_(label, difference_):
    labels_ = tf.div(tf.add(label[:, 0], 1), 2)
    labels_ = tf.reshape(labels_,(-1,1))
    predicts = difference_
    loss_ = tf.losses.log_loss(labels = labels_, predictions = predicts)
    return loss_

def compute_accuracy(prediction, label):
    prediction_ = map(lambda x: [[i, 0][i < 0.5] and [i, 1][i >= 0.5] for i in x], prediction)
    label_ = np.divide(np.add(label[:], 1), 2)
    acc = accuracy_score(label_, prediction_)
    return acc

def compute_accuracy_train(prediction, label):
    prediction_ = map(lambda x: [[i, 0][i < 0.5] and [i, 1][i >= 0.5] for i in x], prediction)
    label_ = np.divide(np.add(label[:], 1), 2)
    acc = accuracy_score(label_, prediction_)
    return acc

def next_batch(s_, e_, inputs, labels_):
    input1_ = inputs[s_:e_, 0]   # 元组的用法，取从s到e这段
    input2_ = inputs[s_:e_, 1]
    y_ = np.reshape(labels_[s_:e_,0], (len(range(s_, e_)), 1))
    return input1_, input2_, y_

def _train_op(total_loss, optimizer, learning_rate, update_gradient_vars):
    # # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    # with tf.control_dependencies([loss_averages_op]):
    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    grads = opt.compute_gradients(total_loss, update_gradient_vars)
    # Apply gradients.
    opt.apply_gradients(grads)
    train_op = tf.no_op(name='train')
    return train_op

def main():
    model_def = 'inception_resnet_v1'
    network = importlib.import_module(model_def)
    gpu_memory_fraction = 0.96
    # Learning params
    FC_SIZE = 1
    learning_rate = 0.0001
    batch_size = 50
    epoch_size = 4
    keep_probability = 1.0
    weight_decay = 0.0

    # Read the file containing the pairs used for testing
    pairs_train, labels_train, _ = read_pairs_path_label_test(image_directory, train_filename)
    pairs_validation, labels_validation, _ = read_pairs_path_label_test(image_directory, validation_filename)
    with tf.Graph().as_default():
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 2), name='image_path')  #
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 2), name='labels')          #

        # read image path
        input_queue = data_flow_ops.FIFOQueue(capacity=300000,
                                              dtypes=[tf.string, tf.int32],
                                              shapes=[(2,), (2,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        images_labels = []
        for _ in range(1):
            filenames, labels = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                # pylint: disable=no-member
                image.set_shape((224, 224, 3))
                images.append(tf.image.per_image_standardization(image))
            images_labels.append([images, labels])

        image_batch, labels_batch = tf.train.batch_join(
            images_labels, batch_size=batch_size_placeholder,
            shapes=[(224, 224, 3), ()], enqueue_many=True,
            capacity=4 * batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the graph
        prelogits, _ = network.inference(image_batch, keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=1,
                                         weight_decay=weight_decay)
        predictions = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        preds_left, preds_right = tf.unstack(tf.reshape(predictions, [-1, 2, 1]), 2, 1)
        labels_1, _ = tf.unstack(tf.reshape(labels_batch, [-1, 2, 1]), 2, 1)
        difference = tf.sigmoid(tf.subtract(preds_left, preds_right))
        loss = log_loss_(labels_1, difference)            #
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
        train_op = _train_op(loss, 'MOM', learning_rate, tf.global_variables())
        sess = tf.Session()
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        with sess.as_default():
            # Training and validation loop
            max_nrof_epochs = 2
            epoch = 0
            while epoch < max_nrof_epochs:
                batch_number = 0
                nrof_pairs = len(pairs_train)
                while batch_number < epoch_size:
                    nrof_batches = int(np.ceil(nrof_pairs * 2 / batch_size))
                    print (nrof_batches)
                    pair_paths = list(itertools.chain(*pairs_train))
                    labels_array = np.reshape(labels_train, (-1, 2))
                    pair_paths_array = np.reshape(np.expand_dims(np.array(pair_paths), 1), (-1, 2))
                    sess.run(enqueue_op,
                             feed_dict={image_paths_placeholder:pair_paths_array, labels_placeholder:labels_array})
                    t_ = sess.run(images_labels)
                    print(np.array(t_).shape)
                    # print(sess.run(filenames))
                    # print(sess.run(labels))
                    nrof_examples = len(pair_paths)
                    i = 0
                    while i < nrof_batches:
                        start_time = time.time()
                        batch_size_ = min(nrof_examples - i * batch_size, batch_size)
                        print(batch_size_)
                        # preds = sess.run(predictions, feed_dict={batch_size_placeholder:batch_size})
                        # err, _ = sess.run([loss, train_op], feed_dict={batch_size_placeholder:batch_size, phase_train_placeholder: True})
                        feed_dict = {batch_size_placeholder: batch_size_, phase_train_placeholder: True}
                        err, _, preds, lab, predictions_, preds_left_, preds_right_ = sess.run([loss, train_op, difference, labels_1, predictions, preds_left, preds_right], feed_dict=feed_dict)
                        duration = time.time() - start_time
                        accuracy = compute_accuracy(preds, lab)
                        print(predictions_.shape)
                        print(preds_left_.shape)
                        print(preds_right_.shape)
                        print('Train-Epoch: [%d]/batch: %d\tTime %.3f\tLoss %2.3f\tAccuracy %2.3f\t' % (
                            epoch, batch_number+1, duration, err, accuracy))
                        i += 1
                        batch_number += 1
                # ### validate ##########
                nro_size = pairs_validation.shape[0]
                labels_array = np.reshape(labels_validation, (-1, 2))
                image_paths_array = np.reshape(np.expand_dims(np.array(pairs_validation), 1), (-1, 2))
                sess.run(enqueue_op, feed_dict={image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
                feed_dict = {batch_size_placeholder: nro_size, phase_train_placeholder: False}
                preds, lab = sess.run([difference, labels_1], feed_dict=feed_dict)
                y = np.reshape(lab, (lab.shape[0], 1))
                accuracy = compute_accuracy(preds, y)
                print('Validation-epoch %d Accuracy validate set %0.2f' % (epoch, accuracy))
                epoch += 1

if __name__ == '__main__':
    main()