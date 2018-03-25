import tensorflow as tf


def distanceFunc(X, Z):
    x_exp = tf.expand_dims(X,1)
    z_exp = tf.expand_dims(tf.transpose(Z), 0)
    dis = tf.reduce_sum((x_exp - z_exp) ** 2, 1)
    return dis
