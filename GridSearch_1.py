# -*- coding: utf-8 -*-
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
    data = np.genfromtxt(f_vector_csv,delimiter=',',dtype=float)
    temp_val,  temp = [], []
    pairs = []
    labels = []
    x1, x2 = -1, -1
    for k in reader:
        temp_val.append(k)
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
        temp = [data[x1],data[x2]]
        pairs.append(temp)
        labels.append(flag)
    return np.array(pairs), np.array(labels)  # 返回的两个值此时都是元组


def ss_net(x,size):
    weights = []
    fc1 = fc_layer(x, 4096, "fc1")
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1, size, "fc2")
    ac2 = tf.nn.relu(fc2)
    fc3 = fc_layer(ac2, 1, "fc3")
    return fc3


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
    if len(shape) == 2:
        init_range = np.sqrt(2.0/(shape[0]+shape[1]))
    elif len(shape) == 1:
        init_range = np.sqrt(2.0 / (shape[0] + 0))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def log_loss_(label, difference):
    predicts = difference
    labels = np.divide(np.add(label, 1), 2)
    loss = tf.losses.log_loss(predicts, labels)
    return loss


def compute_accuracy(prediction, label):
    labels = np.divide(np.add(label, 1), 2)
    acc = accuracy_score(labels, prediction)
    # sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    return acc
    # 返回一个float型的得分数据


def next_batch(s_, e_, inputs, labels_):
    input1_ = inputs[s_:e_, 0]   # 元组的用法，取从s到e这段
    input2_ = inputs[s_:e_, 1]
    y_ = np.reshape(labels_[s_:e_], (len(range(s_, e_)), 1))
    return input1_, input2_, y_

with tf.device('/gpu:0'):
    # create training+validate+test pairs of image
    imageList = readImageList(id_txt)
    train_x, train_labels = read_data_create_pairs(imageList, train_safty)
    validate_x, validate_labels = read_data_create_pairs(imageList, validate_safty)
    test_x, test_labels = read_data_create_pairs(imageList, test_safty)

    images_L = tf.placeholder(tf.float32, shape=([None, 4096]), name='L')
    images_R = tf.placeholder(tf.float32, shape=([None, 4096]), name='R')
    labels = tf.placeholder(tf.float32, shape=([None, 1]), name='label')

batch_size = 256
grid, candidate_para = [], []
tuned_parameters = {'size':(4096,1024,256,64,16), 'learning rate':(1e-2,5e-3,1e-4,5e-4)}
for i in range(len(tuned_parameters['size'])):
    for j in range(len(tuned_parameters['learning rate'])):
        temp = [tuned_parameters['size'][i],tuned_parameters['learning rate'][j]]
        candidate_para.append(temp)
        print('----------------------------------------------------------------------------------------------------'
              '----------------------------------temp divided well-------------------------------------------------'
              '----------------------------------------------------------------------------------------------------')

for k in range(len(candidate_para)):
    with tf.device('/gpu:1'):
        with tf.variable_scope("siamese") as scope:
            model1 = ss_net(images_L,candidate_para[k][0])
            scope.reuse_variables()
            model2 = ss_net(images_R,candidate_para[k][0])

        difference = tf.sigmoid(tf.subtract(model2, model1))
        loss = log_loss_(labels, difference)
        optimizer = tf.train.AdamOptimizer(candidate_para[k][1]).minimize(loss)
    print('a--------------------------------------GGGGGGGGGGGGGG----------------------------------------------a')

    # 启动会话-图
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()
        # 循环训练整个样本30次
        for epoch in range(30):
            avg_loss = 0.
            avg_acc = 0.
            total_batch = int(train_x.shape[0] / batch_size)
            start_time = time.time()
            # 对所有的批量batch进行训练
            for i in range(total_batch):
                s = i * batch_size  # s表示当前批
                e = (i + 1) * batch_size  # e表示下一批
                # Fit training using batch data
                input1, input2, y = next_batch(s, e, train_x, train_labels)
                _, loss_value, predict = sess.run([optimizer, loss, difference],
                                                  feed_dict={images_L: input1, images_R: input2, labels: y})
                feature1 = model1.eval(feed_dict={images_L: input1})
                feature2 = model2.eval(feed_dict={images_R: input2})
                tr_acc = compute_accuracy(predict, y)
                if math.isnan(tr_acc) and epoch != 0:
                    print('tr_acc %0.2f' % tr_acc)
                    pdb.set_trace()
                avg_loss += loss_value
                avg_acc += tr_acc * 100
                print('loss_valuet: %0.2f,  tr_acc: %0.2f' % (loss_value, tr_acc))
            duration = time.time() - start_time
            print('epoch %d time: %f loss: %0.5f acc: %0.2f' % (
            epoch, duration, avg_loss / (total_batch), avg_acc / total_batch))
        y = np.reshape(train_labels, (train_labels.shape[0], 1))
        predict = difference.eval(feed_dict={images_L: train_x[:, 0], images_R: train_x[:, 1]})
        tr_acc = compute_accuracy(predict, y)
        print('%d Accuract training set %0.2f' % (k, 100 * tr_acc))

        # Validate model
        predict = difference.eval(feed_dict={images_L: validate_x[:, 0], images_R: validate_x[:, 1]})
        y = np.reshape(validate_labels, (validate_labels.shape[0], 1))
        vl_acc = compute_accuracy(predict, y)
        print('%d Accuract validate set %0.2f' % (k, 100 * vl_acc))

        # Test model
        predict = difference.eval(feed_dict={images_L: test_x[:, 0], images_R: test_x[:, 1]})
        y = np.reshape(test_labels, (test_labels.shape[0], 1))
        te_acc = compute_accuracy(predict, y)
        print('%d Accuract test set %0.2f' % (k, 100 * te_acc))
    tmp_para = 100*((tr_acc+vl_acc+te_acc)/3)
    grid.append(tmp_para)
print '准确率最好的是 '+str(max(grid))
par = grid.index(max(grid))
print '对应准确率最好的一组参数是 '+str(candidate_para[par])

