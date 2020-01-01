# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple MNIST classifier example with JIT XLA and timelines.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy as np

import input_data
from tensorflow.python.client import timeline
import time
FLAGS = None

from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  input_x = tf.placeholder(tf.float32, [None, 784])
  # Define loss and optimizer
  input_y = tf.placeholder(tf.int64, [None])
  onehot_y = tf.one_hot(input_y,10)
  #
  h_shape = [64, 10]
  h_elem = 64*10
  input_h = tf.placeholder(tf.float32, h_shape)
  # ##########################################################
  init_w = tf.truncated_normal_initializer(stddev=0.1, seed=1)
  init_b = tf.truncated_normal_initializer(stddev=0.1, seed=1)

  w1 = tf.get_variable("w1",[784,128],initializer=init_w)
  b1 = tf.get_variable("b1",[128],initializer=init_b)

  w2 = tf.get_variable("w2",[128,64],initializer=init_w)
  b2 = tf.get_variable("b2",[64],initializer=init_b)

  w3 = tf.get_variable("w3",[64,10],initializer=init_w)
  b3 = tf.get_variable("b3",[10],initializer=init_b)

  wh1 = tf.get_variable("wh1",[64 + 10 + 10 + h_elem, 512],initializer=init_w) #input_x,onehot_y,act_2
  bh1 = tf.get_variable("bh1",[512],initializer=init_b)

  wh2 = tf.get_variable("wh2",[512,512],initializer=init_w) #input_x,onehot_y,act_2
  bh2 = tf.get_variable("bh2",[512],initializer=init_b)

  wh3 = tf.get_variable("wh3",[512,h_elem],initializer=init_w) #input_x,onehot_y,act_2
  bh3 = tf.get_variable("bh3",[h_elem],initializer=init_b)
  ##step 1
  t_hw = input_h
  #
  net_1 = tf.matmul(input_x, w1) + b1
  act_1 = tf.nn.relu(net_1)

  net_2 = tf.matmul(act_1, w2) + b2
  act_2 = tf.nn.relu(net_2)

  net3 = tf.matmul(act_2, w3) + b3
  net4 = tf.matmul(act_2, t_hw)
  output_y_1 = tf.nn.softmax(net3 + net4)

  loss_1 = tf.reduce_mean((onehot_y - output_y_1) * (onehot_y - output_y_1))
  ##
  b_t_hw = tf.tile(tf.reshape(t_hw,[1,h_elem]), [tf.shape(input_x)[0],1])
  net6 = tf.stop_gradient(tf.concat([act_2, onehot_y - output_y_1, net3, b_t_hw],axis=1))
  print(net6)

  net7 = tf.matmul(net6, wh1) + bh1
  act_7 = tf.nn.relu(net7)

  net8 = tf.matmul(act_7, wh2) + bh2
  act_8 = tf.nn.relu(net8)

  net9 = tf.matmul(act_8, wh3) + bh3
  net9 = tf.nn.sigmoid(tf.reduce_mean(net9,axis=0)) - 0.5

  t_hw = t_hw + 0.01 * tf.stop_gradient(loss_1) * tf.reshape(net9,h_shape)

  net42_1 = tf.matmul(tf.stop_gradient(act_2), t_hw)
  output_y2_1 = tf.nn.softmax(tf.stop_gradient(net3) + net42_1)

  loss2_1 = tf.reduce_mean((onehot_y - output_y2_1) * (onehot_y - output_y2_1))
  ##
  ##step2
  loss_list = [loss2_1]
  output_list = [output_y2_1]
  hidden_list = [net42_1]
  for _ in range(1):
    b_t_hw_i = tf.tile(tf.reshape(t_hw,[1,h_elem]), [tf.shape(input_x)[0],1])
    net6_i = tf.stop_gradient(tf.concat([act_2, onehot_y - output_list[-1], net3, b_t_hw_i],axis=1))
    
    net7_i = tf.matmul(net6_i, wh1) + bh1
    act_7_i = tf.nn.relu(net7_i)

    net8_i = tf.matmul(act_7_i, wh2) + bh2
    act_8_i = tf.nn.relu(net8_i)

    net9_i = tf.matmul(act_8_i, wh3) + bh3
    net9_i = tf.nn.sigmoid(tf.reduce_mean(net9_i,axis=0)) - 0.5

    t_hw = t_hw + 0.01 * tf.stop_gradient(loss_list[-1]) * tf.reshape(net9_i,h_shape)

    net42_i = tf.matmul(tf.stop_gradient(act_2), t_hw)
    output_y2_i = tf.nn.softmax(tf.stop_gradient(net3) + net42_i)

    loss2_i = tf.reduce_mean((onehot_y - output_y2_i) * (onehot_y - output_y2_i))

    loss_list.append(loss2_i)
    output_list.append(output_y2_i)
    hidden_list.append(net42_i)
  
  # t_hw = tf.Print(t_hw,[t_hw,loss1,loss_list[-1]],message="debug:") #loss1,loss_list[-1]

  floss_learn = -(loss_list[0] - loss_list[-1])
  floss_all = loss_1 + loss_list[-1]
  floss_task = loss_1
  output_h = t_hw
  output_y = output_list[-1]
  ##########################################################
  optimizer1 = tf.train.GradientDescentOptimizer(0.01)
  gd1 = optimizer1.compute_gradients(floss_learn)
  keep_list = ["wh1:0","bh1:0","wh2:0","bh2:0","wh3:0","bh3:0"]
  new_gd1 = []
  for item in gd1:
    if item[1].name in keep_list:
      new_gd1.append(item)
  train_step_learner = optimizer1.apply_gradients(gd1)

  optimizer2 = tf.train.GradientDescentOptimizer(0.01)
  gd2 = optimizer2.compute_gradients(floss_all)
  train_step_all = optimizer2.apply_gradients(gd2)

  optimizer3 = tf.train.GradientDescentOptimizer(0.01)
  gd3 = optimizer3.compute_gradients(floss_task)
  keep_list = ["w1:0","b1:0","w2:0","b2:0","w3:0","b3:0"]
  new_gd3 = []
  for item in gd3:
    if item[1].name in keep_list:
      new_gd3.append(item)
  train_step_task = optimizer3.apply_gradients(new_gd3)

  sess = tf.Session()
  tf.global_variables_initializer().run(session=sess)
  # Train

  train_loops = 10000
  g_mid_h = np.zeros(h_shape,dtype=np.float32)
  for i in range(train_loops):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if i % 2 == 0:
      pass
      _, g_mid_h = sess.run([train_step_learner, output_h], feed_dict={input_x: batch_xs, input_y: batch_ys, input_h: g_mid_h})
    else:
      pass
      _, g_mid_h = sess.run([train_step_task, output_h], feed_dict={input_x: batch_xs, input_y: batch_ys, input_h: g_mid_h})

    if i % 100 == 0:
      # Test trained model
      print("begin")
      for _ in range(5):
        correct_prediction = tf.equal(tf.argmax(output_y, 1), input_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        res,g_mid_h = sess.run([accuracy,output_h],
                      feed_dict={input_x: mnist.test.images,
                                  input_y: mnist.test.labels,
                                  input_h: g_mid_h})

        print(res)
      print("end")

  sess.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='.',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
