import tensorflow as tf
import tensorflow.contrib.layers as layers

class Model:
    def __init__(self):
        observation = tf.placeholder(tf.float32, [None, 7, 7, 1])

        h = layers.convolution2d(observation, 32, [4, 4], activation_fn=tf.nn.elu)
        h = layers.flatten(h)
        h_mu = layers.fully_connected(h, 10, activation_fn=None)
        h_sigma = layers.fully_connected(h, 10, activation_fn=None)
        z = h_mu + tf * h_sigma

    