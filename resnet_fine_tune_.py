# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Flatten
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import accuracy_score
from datetime import datetime
import os.path
import time
import importlib
import math,pdb
import tensorflow as tf
import resnet_50_

from tensorflow.python.ops import data_flow_ops

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 模型保存地址
model_dir = '/storage/guoyangyang/ziwen/resnet50_finetune/models/model.ckpt'
log_dir = '/storage/guoyangyang/ziwen/resnet50_finetune/logs/model.ckpt'
#
# image_directory = '/storage/guoyangyang/ziwen/feature_ext/Places Pulse/'
# train_filename = '/storage/guoyangyang/ziwen/Ranking_network/votes_safety/train_sa.csv'
# validation_filename = '/storage/guoyangyang/ziwen/Ranking_network/votes_safety/validate_sa.csv'
image_directory = '/home/czw/图片/Places Pulse/'
train_filename = '/home/czw/文档/csv/safety/test_train.csv'
validation_filename = '/home/czw/文档/csv/safety/test_validation.csv'

def read_pairs_path_label(image_directory, csv_filename):  # return three lists of pairs of path of image and labels
    pairs = []
    left = []
    right = []
    labels = []
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
        left.append(x_path)
        right.append(y_path)
        labels.append(flag)
    return np.array(left), np.array(right), np.array(labels)

def read_image_path(filenames):
    images = []
    print('read_image_path is working')
    for filename in tf.unstack(filenames):
        # print(filename)
        file_contents = tf.read_file(filename[0])
        image = tf.image.decode_jpeg(file_contents, channels=3)
        image.set_shape((224, 224, 3))
        images.append(image)
    # print(np.array(images).shape)
    images = tf.identity(images, 'images')
    return images

def log_loss_(label, difference_):
    predicts = difference_
    labels_ = tf.div(tf.add(label, 1), 2)
    loss_ = tf.losses.log_loss(labels = labels_, predictions = predicts)
    return loss_

def compute_accuracy(prediction, label):
    prediction_ = map(lambda x: [[i, 0][i < 0.5] and [i, 1][i >= 0.5] for i in x], prediction)
    label_ = np.divide(np.add(label, 1), 2)
    acc = accuracy_score(label_, prediction_)
    return acc

def next_batch(s_, e_, inputs_x, inputs_y, labels_):
    input1_ = inputs_x[s_:e_]   # 元组的用法，取从s到e这段
    input2_ = inputs_y[s_:e_]
    input1_ = np.reshape(input1_,(len(input1_),1))
    input2_ = np.reshape(input2_,(len(input2_),1))
    y_ = np.reshape(labels_[s_:e_], (len(range(s_, e_)), 1))
    return input1_, input2_, y_

def train_op_(total_loss, optimizer, learning_rate, update_gradient_vars):
    # Generate moving averages of all losses and associated summaries.
    # Compute gradients.
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
    apply_gradient_op = opt.apply_gradients(grads)
    train_op = tf.no_op(name='train')
    return train_op

def main():
    model_def = 'inception_resnet_v1'
    network = importlib.import_module(model_def)
    gpu_memory_fraction = 0.96
    FC_SIZE = 1
    learning_rate = 0.0001
    batch_size = 5
    keep_probability = 1.0
    weight_decay = 0.0
    global_step = tf.Variable(0, trainable=False)

    # List of trainable variables of the layers we want to train
    # var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Read the file containing the pairs used for testing
    x_train, y_train, labels_train = read_pairs_path_label(image_directory, train_filename)
    x_validation, y_validation, labels_validation = read_pairs_path_label(image_directory, validation_filename)
    with tf.Graph().as_default():
        image_x_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_x')  #
        image_y_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_y')  #
        image_placeholder = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')          #
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        # read image path
        image_pair_path = tf.stack([image_x_placeholder, image_y_placeholder], axis=1)
        ## train
        image_path_reshape = tf.reshape(image_pair_path, (batch_size*2, 1))
        image_total = read_image_path(image_path_reshape)
        ## validate
        batch_size_val = x_validation.shape[0]
        image_path_reshape_val = tf.reshape(image_pair_path, (batch_size_val * 2, 1))
        image_total_val = read_image_path(image_path_reshape_val)

        # Build the graph
        # model = resnet_50_.ResNet50(include_top=False, weights='imagenet')
        # x = model.output
        # x = GlobalAveragePooling2D()(x)
        # predictions = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
        # predictions = Dense(1, activation='softmax')(predictions)  # new softmax layer
        # model_ = Model(inputs=model.input, outputs=predictions)
        # preds = model_.predict(image_total)

        prelogits, _ = network.inference(image_placeholder, keep_probability,phase_train=phase_train_placeholder, bottleneck_layer_size=1)

        preds = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        preds_reshape = tf.reshape(preds,(batch_size_placeholder, 2, 1))
        preds_left, preds_right = tf.unstack(preds_reshape, num=2, axis=1)
        difference = tf.sigmoid(tf.subtract(preds_left, preds_right))
        # print(difference)
        loss = log_loss_(labels_placeholder, difference)            #
        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
        train_op = train_op_(loss, 'MOM', learning_rate, tf.global_variables())
        # Start running operations on the Graph.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess = tf.Session()
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with sess.as_default():
            # Training and validation loop
            for epoch in range(3):
                avg_loss = 0.
                avg_acc = 0.
                nrof_batch = int(x_train.shape[0] / batch_size)
                print('Working...')
                start_time = time.time()
                # 对所有的批量batch进行训练
                for i in range(nrof_batch):
                    s = i * batch_size  # s表示当前批
                    e = (i + 1) * batch_size  # e表示下一批
                    # Fit training using batch data
                    left, right, y = next_batch(s, e, x_train, y_train, labels_train)
                    num1 = left.shape[0]
                    # print(left)
                    # feed_dict = {image_x_placeholder:input1_path, image_y_placeholder:input2_path, labels_placeholder:y}
                    feed_dict = {image_x_placeholder:left, image_y_placeholder: right, labels_placeholder: y,
                                 batch_size_placeholder: num1,phase_train_placeholder: True}
                    image_total_ = sess.run(image_total,feed_dict=feed_dict)
                    # print (image_total)
                    feed_dict = {image_placeholder: image_total_, labels_placeholder: y,
                                 batch_size_placeholder: num1, phase_train_placeholder: True}
                    print (y)
                    _, loss_value, predict = sess.run([train_op, loss, difference],feed_dict=feed_dict)
                    tr_acc = compute_accuracy(predict, y)
                    if math.isnan(tr_acc) and epoch != 0:
                        print('tr_acc %0.2f' % tr_acc)
                        pdb.set_trace()
                    avg_loss += loss_value
                    avg_acc += tr_acc * 100
                duration = time.time() - start_time
                print('epoch %d  time: %f loss %0.5f acc %0.2f' % (
                    epoch, duration, avg_loss / nrof_batch, avg_acc / nrof_batch))

                x_validation = np.reshape(x_validation, (x_validation.shape[0],1))
                y_validation = np.reshape(y_validation, (y_validation.shape[0], 1))
                labels_validation = np.reshape(labels_validation, (labels_validation.shape[0], 1))
                feed_dict_ = {image_x_placeholder: x_validation[:], image_y_placeholder: y_validation[:],batch_size_placeholder: len(x_validation)}
                image_total_v = sess.run(image_total_val, feed_dict=feed_dict_)
                feed_dict_ = {image_placeholder:image_total_v, labels_placeholder:labels_validation, batch_size_placeholder:len(x_validation),phase_train_placeholder:False}
                predict_va = sess.run(difference, feed_dict=feed_dict_)
                y = np.reshape(labels_validation, (labels_validation.shape[0], 1))
                vl_acc = compute_accuracy(predict_va, y)
                print('epoch %d Accuracy validate set %0.2f' % (epoch, 100 * vl_acc))

if __name__ == '__main__':
    main()