# -*- coding: utf-8 -*-
import numpy as np
import time
import tensorflow as tf
import math
import csv
import pdb
import os
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_safty = r'/storage/guoyangyang/ziwen/Ranking_network/votes_safety/train_safety.csv'
validate_safty = r'/storage/guoyangyang/ziwen/Ranking_network/votes_safety/validate_safety.csv'
test_safty = r'/storage/guoyangyang/ziwen/Ranking_network/votes_safety/test_safety.csv'
id_txt = r'/storage/guoyangyang/ziwen/feature_ext/input_imagelist.txt'
f_vector_csv = r'/storage/guoyangyang/ziwen/feature_ext/feature_extracted.csv'


# 读取txt文件
def readImageList(input_imagelist):
    imageList_ = []
    with open(input_imagelist, 'r') as fi:
        while(True):
            line = fi.readline().strip().split()  # every line is a image file name
            if not line:
                break
            imageList_.append(line[0].rstrip('.jpg'))
    return imageList_


# 序列号匹配对应的特征向量，并返回一个array数组(包含对应特征向量及标签)
def read_data_create_pairs(imageList_, safty):
    f = open(safty)
    reader = csv.reader(f)
    header = next(reader)
    f_f = open(f_vector_csv)
    reader_f = csv.reader(f_f)
    temp_val, temp_f, temp = [], [], []
    pairs = []
    labels = []
    x1, x2 = -1, -1
    for k in reader:
        temp_val.append(k)
    for k in reader_f:
        temp_f.append(k)
    for i in range(len(temp_val)):
        if temp_val[i][2] == 'left':
            flag = 1
        elif temp_val[i][2] == 'right':
            flag = -1
        else:
            continue
        for j in range(len(imageList_)):
            if temp_val[i][0] == imageList_[j]:
                x1 = j
            if temp_val[i][1] == imageList_[j]:
                x2 = j
                if x1 != -1:
                    break
        temp = [temp_f[x1],temp_f[x2]]
        pairs.append(temp)
        labels.append(flag)
    return np.array(pairs), np.array(labels)  # 返回的两个值此时都是元组


def ss_net(x):
    weights = []
    fc1 = fc_layer(x, 4096, "fc1")
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1, 4096, "fc2")
    ac2 = tf.nn.relu(fc2)
    fc3 = fc_layer(ac2, 1, "fc3")
    return fc3


def fc_layer(bottom, n_weight, name):   # 注意bottom是256×4096的矩阵
    assert len(bottom.get_shape()) == 2     # 只有tensor有这个方法， 返回是一个tuple
    n_prev_weight = bottom.get_shape()[1]   # bottom.get_shape() 即 （256, 4096）
    initer = tf.truncated_normal_initializer(stddev=0.01)
    # 截断正太分布 均值mean（=0）,标准差stddev,只保留[mean-2*stddev,mean+2*stddev]内的随机数
    W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
    b = tf.get_variable(name + 'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)  # tf.nn.bias_add(value, bias, name = None) 将偏置项b加到values上
    return fc


def log_loss_(label, difference_):
    predicts = difference_
    labels_ = tf.div(tf.add(label, 1), 2)
    loss_ = tf.losses.log_loss(labels = labels_, predictions = predicts)
    return loss_


# def compute_accuracy_train(prediction, labels):
#     labels = tf.div(tf.add(labels, 1), 2)
#     return labels[prediction.ravel() < 0.5].mean()


def compute_accuracy(prediction, label):
    prediction_ = map(lambda x: [[i, 0][i < 0.5] and [i, 1][i >= 0.5] for i in x], prediction)
    label_ = np.divide(np.add(label, 1), 2)
    acc = accuracy_score(label_, prediction_)
    # sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    return acc
    # 返回一个float型的得分数据


def next_batch(s_, e_, inputs, labels_):
    input1_ = inputs[s_:e_, 0]   # 元组的用法，取从s到e这段
    input2_ = inputs[s_:e_, 1]
    y_ = np.reshape(labels_[s_:e_], (len(range(s_, e_)), 1))
    return input1_, input2_, y_

batch_size = 512
# create training+validate+test pairs of image
imageList = readImageList(id_txt)
train_x, train_labels = read_data_create_pairs(imageList, train_safty)
validate_x, validate_labels = read_data_create_pairs(imageList, validate_safty)
test_x, test_labels = read_data_create_pairs(imageList, test_safty)

images_L = tf.placeholder(tf.float32, shape=([None, 4096]), name='L')
images_R = tf.placeholder(tf.float32, shape=([None, 4096]), name='R')
labels = tf.placeholder(tf.float32, shape=([None, 1]), name='label')

with tf.variable_scope("siamese") as scope:
    model1 = ss_net(images_L)
    scope.reuse_variables()
    model2 = ss_net(images_R)

difference = tf.sigmoid(tf.subtract(model2, model1))
loss = log_loss_(labels, difference)
optimizer = tf.train.MomentumOptimizer(1e-3, 0.9).minimize(loss)
print('a------------------------------------******------------------------------------------a')
# 启动会话-图
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # 初始化所有变量
    tf.global_variables_initializer().run()
    # 循环训练整个样本30次
    for epoch in range(20):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(train_x.shape[0] / batch_size)
        start_time = time.time()
        # 对所有的批量batch进行训练
        for i in range(total_batch):
            s = i * batch_size   # s表示当前批
            e = (i + 1) * batch_size  # e表示下一批
            # Fit training using batch data
            input1, input2, y = next_batch(s, e, train_x, train_labels)
            _, loss_value, predict = sess.run([optimizer, loss, difference],feed_dict={images_L: input1, images_R: input2, labels: y})
            feature1 = model1.eval(feed_dict={images_L: input1})
            feature2 = model2.eval(feed_dict={images_R: input2})
            tr_acc = compute_accuracy(predict, y)
            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
                pdb.set_trace()
            avg_loss += loss_value
            avg_acc += tr_acc * 100
            # print('loss_valuet: %0.2f,  ar_acc: %0.2f' % (loss_value, tr_acc))
        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' % (epoch, duration, avg_loss / (total_batch), avg_acc / total_batch))
    y = np.reshape(train_labels, (train_labels.shape[0], 1))
    predict = difference.eval(feed_dict={images_L: train_x[:, 0], images_R: train_x[:, 1]})
    tr_acc = compute_accuracy(predict, y)
    print('Accuract training set %0.2f' % (100 * tr_acc))

    # Validate model
    predict = difference.eval(feed_dict={images_L: validate_x[:, 0], images_R: validate_x[:, 1]})
    y = np.reshape(validate_labels, (validate_labels.shape[0], 1))
    vl_acc = compute_accuracy(predict, y)
    print('Accuract validate set %0.2f' % (100 * vl_acc))

    # Test model
    predict = difference.eval(feed_dict={images_L: test_x[:, 0], images_R: test_x[:, 1]})
    y = np.reshape(test_labels, (test_labels.shape[0], 1))
    te_acc = compute_accuracy(predict, y)
    print('Accuract test set %0.2f' % (100 * te_acc))