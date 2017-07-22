# -*- coding: utf-8 -*-
''''
gpu
with tf.device('/cpu:0')
'''

import numpy as np
import time
import tensorflow as tf
import math
import csv
import pdb
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train_safty = r'/storage/guoyangyang/ziwen/Ranking_network/votes_safety/train_safety.csv'
validate_safty = r'/storage/guoyangyang/ziwen/Ranking_network/votes_safety/validate_safety.csv'
test_safty = r'/storage/guoyangyang/ziwen/Ranking_network/votes_safety/test_safety.csv'
id_txt = r'/storage/guoyangyang/ziwen/feature_ext/input_imagelist.txt'
f_vector_csv = r'/storage/guoyangyang/ziwen/feature_ext/feature_extracted.csv'

# Processing Units logs
log_device_placement = True


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
            flag = 0
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


def ss_net(x, dropout_ratio):
    # weights = []
    fc1 = fc_layer(x, 1024, "fc1")
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1, 1024, "fc2")
    mean1, variance1 = tf.nn.moments(fc2, [0,1,2])
    bn_fc2 = tf.nn.batch_normalization(fc2, mean=mean1, variance=variance1)
    ac2 = tf.nn.relu(bn_fc2)
    dc2 = tf.nn.dropout(ac2, 1-dropout_ratio)
    fc3 = fc_layer(dc2, 1, "fc3")
    dc3 = tf.nn.dropout(fc3, 1-dropout_ratio)
    return dc3


def fc_layer(bottom, n_weight, name):   # 注意bottom是256×4096的矩阵
    assert len(bottom.get_shape()) == 2     # 只有tensor有这个方法， 返回是一个tuple
    n_prev_weight = bottom.get_shape()[1]   # bottom.get_shape() 即 （256, 4096）
    # initer = tf.truncated_normal_initializer(stddev=0.01)
    # # 截断正太分布 均值mean（=0）,标准差stddev,只保留[mean-2*stddev,mean+2*stddev]内的随机数
    # W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
    # b = tf.get_variable(name + 'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
    W = glorot(shape=[n_prev_weight, n_weight], name = name + 'W')
    b = glorot(shape=[n_weight], name = name + 'b')
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)  # tf.nn.bias_add(value, bias, name = None) 将偏置项b加到values上
    return fc


def glorot(shape, name=None):
    init_range = np.sqrt(2.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def log_loss_(label, difference):
    predicts = difference
    loss = tf.losses.log_loss(predicts, label)
    return loss


def compute_accuracy(prediction, label):
    acc = accuracy_score(labels, prediction)
    # sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    return acc
    # 返回一个float型的得分数据


def next_batch(s_, e_, inputs, labels_):
    input1_ = inputs[s_:e_, 0]   # 元组的用法，取从s到e这段
    input2_ = inputs[s_:e_, 1]
    y_ = np.reshape(labels_[s_:e_], (len(range(s_, e_)), 1))
    return input1_, input2_, y_
# 初始化所有变量
init = tf.global_variables_initializer()
batch_size = 256

# create training+validate+test pairs of image
imageList = readImageList(id_txt)
train_x, train_labels = read_data_create_pairs(imageList, train_safty)
validate_x, validate_labels = read_data_create_pairs(imageList, validate_safty)
test_x, test_labels = read_data_create_pairs(imageList, test_safty)
#####################################################

X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
#####################################################
with tf.device('/gpu:0'):
    images_L = tf.placeholder(tf.float32, shape=([None, 4096]), name='L')
    images_R = tf.placeholder(tf.float32, shape=([None, 4096]), name='R')
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='label')

    with tf.variable_scope("siamese") as scope:
        model1 = ss_net(images_L)
        scope.reuse_variables()
        model2 = ss_net(images_R)
    # # GridSearch_0 调整 Ranknet_1的hyperparameter,搜索两个parameters
    # parameters = {'size':(4096,1024,256,64,16),'learning rate':(1e-2,5e-3,1e-4,5e-4),'dropout_ratio':(0.0,0.1,0.2,0.3)}
    # clf = GridSearchCV(parameters)
    # clf.fit(images_L, images_R)

with tf.device('/gpu:1'):
    difference = tf.sigmoid(tf.subtract(model2, model1))
    loss = log_loss_(labels, difference)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

# 启动会话-图
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    tf.global_variables_initializer().run()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # 循环训练整个样本30次
    for epoch in range(30):
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
            _,loss_value,predict = sess.run([optimizer,loss,difference],feed_dict={images_L:input1,images_R:input2,labels:y})
            feature1 = model1.eval(feed_dict={images_L: input1})
            feature2 = model2.eval(feed_dict={images_R: input2})
            tr_acc = compute_accuracy(predict, y)
            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
                pdb.set_trace()
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        # print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
        duration = time.time() - start_time
        print('epoch %d time: %f loss %0.5f acc %0.2f' % (epoch, duration, avg_loss/(total_batch), avg_acc/total_batch))
    y = np.reshape(train_labels, (train_labels.shape[0], 1))
    predict = difference.eval(feed_dict={images_L: train_x[:, 0], images_R: train_x[:, 1], labels: train_labels})
    tr_acc = compute_accuracy(predict, y)
    print('Accuract training set %0.2f' % (100 * tr_acc))

    # Validate model
    predict = difference.eval(feed_dict={images_L:validate_x[:, 0], images_R:validate_x[:, 1], labels:validate_labels})
    y = np.reshape(validate_labels, (validate_labels.shape[0], 1))
    te_acc = compute_accuracy(predict, y)
    print('Accuract validate set %0.2f' % (100 * te_acc))

    # Test model
    predict = difference.eval(feed_dict={images_L: test_x[:, 0], images_R: test_x[:, 1], labels: test_labels})
    y = np.reshape(test_labels, (test_labels.shape[0], 1))
    te_acc = compute_accuracy(predict, y)
    print('Accuract test set %0.2f' % (100 * te_acc))
