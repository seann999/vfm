import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

class Model:
    def __init__(self, frame_size, n_steps):
        self.global_step = tf.Variable(0)
        self.observation = tf.placeholder(tf.float32, [None, n_steps, frame_size, frame_size, 1])
        self.target = tf.placeholder(tf.float32, [None, n_steps, frame_size, frame_size, 1])
        self.true_label = tf.placeholder(tf.int32, [None])
        self.locations = tf.placeholder(tf.float32, [None, n_steps+1, 2])

        observation = tf.reshape(self.observation, [-1, frame_size**2])
        observation = tf.concat([observation, tf.reshape(self.locations[:, :-1, :], [-1, 2])], 1)

        target = tf.reshape(self.target, [-1, frame_size ** 2])
        #h = layers.convolution2d(observation, 32, [4, 4], activation_fn=tf.nn.elu)
        #h = layers.flatten(h)
        h = observation

        h_size = 512
        cell_size = 512
        act = tf.nn.relu

        h = layers.fully_connected(h, h_size, activation_fn=act)
        h = layers.fully_connected(h, h_size, activation_fn=act)

        h = tf.reshape(h, [-1, n_steps, h_size])
        h = tf.unstack(h, n_steps, 1)
        lstm_cell = rnn.BasicLSTMCell(cell_size, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, h, dtype=tf.float32)
        h = tf.stack(outputs, 1)

        h = tf.reshape(h, [-1, cell_size])

        n_z = 10

        z_mean = layers.fully_connected(h, n_z, activation_fn=None)
        z_logvar = layers.fully_connected(h, n_z, activation_fn=None)

        eps = tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        self.z = tf.add(z_mean,
                        tf.multiply(tf.sqrt(tf.exp(z_logvar)), eps))

        self.z_test = layers.fully_connected(tf.stop_gradient(self.z), 10, activation_fn=None)
        self.class_test = tf.nn.softmax(self.z_test)

        locations = tf.reshape(self.locations[:, 1:, :], [-1, 2])
        h = tf.concat([self.z, locations], 1)

        h = layers.fully_connected(h, h_size, activation_fn=act)
        h = layers.fully_connected(h, h_size, activation_fn=act)
        self.prediction = layers.fully_connected(h, frame_size ** 2, activation_fn=None)
        self.prediction2d = tf.reshape(self.prediction, [-1, frame_size, frame_size, 1])

        self.BCE = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=target)
        self.BCE = tf.reduce_sum(self.BCE, axis=1)

        self.avg_BCE_dec = tf.reshape(self.BCE, [-1, n_steps])
        self.avg_BCE_dec = tf.reduce_mean(self.avg_BCE_dec[:, :-1] - self.avg_BCE_dec[:, 1:])

        self.BCE = tf.reduce_mean(self.BCE)

        KLD = -0.5 * (1 + z_logvar - tf.pow(z_mean, 2) - tf.exp(z_logvar))
        KLD = tf.reduce_mean(tf.reduce_sum(KLD, axis=1))

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.z_test, labels=self.true_label)

        self.avg_ce_dec = tf.reshape(class_loss, [-1, n_steps])
        self.avg_ce_dec = tf.reduce_mean(self.avg_ce_dec[:, :-1] - self.avg_ce_dec[:, 1:])
        self.class_acc = tf.reduce_mean(tf.cast(tf.equal(self.true_label, tf.cast(tf.argmax(self.z_test, 1), tf.int32)), tf.float32))

        self.loss = KLD + self.BCE + tf.reduce_mean(class_loss)
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss, global_step=self.global_step)