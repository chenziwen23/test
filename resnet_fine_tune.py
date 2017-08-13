# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import data_flow_ops

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 模型保存地址
model_dir = '/storage/guoyangyang/ziwen/resnet50_finetune/models/model.ckpt'
log_dir = '/storage/guoyangyang/ziwen/resnet50_finetune/logs/model.ckpt'
image_directory = '/storage/guoyangyang/ziwen/feature_ext/Places Pulse/'
train_filename = '/storage/guoyangyang/ziwen/Ranking_network/votes_safety/train_sa.csv'
validation_filename = '/storage/guoyangyang/ziwen/Ranking_network/votes_safety/validate_sa.csv'

def read_pairs_path_label(image_directory, csv_filename):  # return two lists of pairs of path of image and labels
    pairs = []
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
        pair = [x_path, y_path]
        pairs.append(pair)
        labels.append(flag)
    return np.array(pairs), np.array(labels)

def read_image_path(filename):
    file_contents = tf.read_file(filename)
    image = tf.image.decode_image(file_contents, channels=3)
    return image

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

def next_batch(s_, e_, inputs, labels_):
    input1_ = inputs[s_:e_, 0]   # 元组的用法，取从s到e这段
    input2_ = inputs[s_:e_, 1]
    y_ = np.reshape(labels_[s_:e_], (len(range(s_, e_)), 1))
    return input1_, input2_, y_

def main():
    saver = tf.train.import_meta_graph('models/ResNet-L50.meta')
    with tf.Session() as sess:
        network = saver.restore(sess,"models/ResNet-L50.ckpt")

    # Read the file containing the pairs used for testing
    pairs_train, labels_train = read_pairs_path_label(image_directory, train_filename)
    pairs_validation, labels_validation = read_pairs_path_label(image_directory, validation_filename)
    with tf.Graph().as_default():
        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_x_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_x')  #
        image_y_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_y')  #
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')          #

        # Build the graph
        out_net = network.layers['pool5']
        net = tf.contrib.layers.fully_connected(out_net, 1, activation_fn=tf.nn.relu,
                                          weights_initializer= tf.truncated_normal_initializer(stddev=0.01),
                                          biases_initializer=tf.constant(0.01))
        with tf.variable_scope("siamese") as scope:
            model1 = net(image_x_placeholder)
            scope.reuse_variables()
            model2 = net(image_y_placeholder)

        difference = tf.sigmoid(tf.subtract(model1, model2))
        loss = log_loss_(labels_placeholder, difference)            #
        optimizer = tf.train.MomentumOptimizer(1e-5, 0.9).minimize(loss)

        # # Create a saver
        # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=2)     #

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with sess.as_default():
            # Training and validation loop
            for epoch in range(100):
                avg_loss = 0.
                avg_acc = 0.
                total_batch = int(train_x.shape[0] / batch_size)
                start_time = time.time()
                # 对所有的批量batch进行训练
                for i in range(total_batch):
                    s = i * batch_size  # s表示当前批
                    e = (i + 1) * batch_size  # e表示下一批
                    # Fit training using batch data
                    input1_path, input2_path, y = next_batch(s, e, pairs_train, labels_train)
                    # read image path
                    input1 = read_image_path(input1_path)
                    input2 = read_image_path(input2_path)
                    _, loss_value, predict = sess.run([optimizer, loss, difference],
                                                      feed_dict={images_L: input1, images_R: input2, labels: y})
                    tr_acc = compute_accuracy(predict, y)
                    if math.isnan(tr_acc) and epoch != 0:
                        print('tr_acc %0.2f' % tr_acc)
                        pdb.set_trace()
                    avg_loss += loss_value
                    avg_acc += tr_acc * 100
                duration = time.time() - start_time
                print('epoch %d  time: %f loss %0.5f acc %0.2f' % (
                    epoch, duration, avg_loss / total_batch, avg_acc / total_batch))

                validate_x = read_image_path(pairs_validation[:, 0])
                validate_y = read_image_path(pairs_validation[:, 1])
                predict_va = difference.eval(feed_dict={images_L: validate_x, images_R: validate_y})
                y = np.reshape(labels_validation, (labels_validation.shape[0], 1))
                vl_acc = compute_accuracy(predict_va, y)
                print('epoch %d Accuracy validate set %0.2f' % (epoch, 100 * vl_acc))
                # Train for one epoch

            # # Save variables and the metagraph if it doesn't exist already
            # save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    return model_dir

if __name__ == '__main__':
    main()