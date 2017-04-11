import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

class Model:
    def __init__(self, frame_size, n_steps):
        self.observation = tf.placeholder(tf.float32, [None, n_steps, frame_size, frame_size, 1])
        self.target = tf.placeholder(tf.float32, [None, n_steps, frame_size, frame_size, 1])
        self.locations = tf.placeholder(tf.float32, [None, n_steps, 2])

        observation = tf.reshape(self.observation, [-1, frame_size**2])
        target = tf.reshape(self.target, [-1, frame_size ** 2])
        #h = layers.convolution2d(observation, 32, [4, 4], activation_fn=tf.nn.elu)
        #h = layers.flatten(h)
        h = observation

        h_size = 128
        cell_size = 128
        act = tf.nn.relu

        h = layers.fully_connected(h, h_size, activation_fn=act)
        h = layers.fully_connected(h, h_size, activation_fn=act)

        h = tf.reshape(h, [-1, n_steps, h_size])
        h = tf.unstack(h, n_steps, 1)
        lstm_cell = rnn.BasicLSTMCell(cell_size, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, h, dtype=tf.float32)
        h = tf.stack(outputs, 1)

        h = tf.reshape(h, [-1, cell_size])

        n_z = 32

        z_mean = layers.fully_connected(h, n_z, activation_fn=None)
        z_logvar = layers.fully_connected(h, n_z, activation_fn=None)

        eps = tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        self.z = tf.add(z_mean,
                        tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps))

        #self.z_test = layers.fully_connected(tf.stop_gradient(h), frame_size**2, activation_fn=None)
        #self.class_test =

        locations = tf.reshape(self.locations, [-1, 2])
        h = tf.concat([self.z, locations], 1)

        h = layers.fully_connected(h, h_size, activation_fn=act)
        h = layers.fully_connected(h, h_size, activation_fn=act)
        self.reconstruction = layers.fully_connected(h, frame_size**2, activation_fn=None)
        self.reconstruction2d = tf.reshape(self.reconstruction, [-1, frame_size, frame_size, 1])

        BCE = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.reconstruction, labels=target)
        KLD = -0.5 * (1 + z_logvar - tf.pow(z_mean, 2) - tf.exp(z_logvar))

        BCE = tf.reduce_sum(BCE, axis=1)
        KLD = tf.reduce_sum(KLD, axis=1)

        self.loss = tf.reduce_mean(BCE + KLD)
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)