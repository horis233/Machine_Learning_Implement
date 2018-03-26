import tensorflow as tf
import numpy as np

def buildGraph(input,num_hidden):
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=3.0/(input.get_shape()[0])+num_hidden), name='weights')
    b = tf.Variable(tf.zeros([10]))

    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    # y_target = tf.placeholder(tf.float32, [None, 10], name='target_y')

    output = tf.matmul(X, W) + b
    output = tf.nn.relu(output)
    return output