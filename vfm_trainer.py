from tensorflow.examples.tutorials.mnist import input_data
import cv2
import vfm_model
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("data", False)

frame = 14
steps = 10
batch_size = 32

model = vfm_model.Model(frame, steps)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cv2.namedWindow("images", cv2.WINDOW_NORMAL)
cv2.namedWindow("gen", cv2.WINDOW_NORMAL)

for i in range(10000000):
    images, labels = mnist.train.next_batch(batch_size)

    instances = []
    locations = []

    for image in images:
        glimpses = []
        image = np.reshape(image, [28, 28])

        for k in range(steps+1):
            x, y = np.random.randint(low=0, high=28-frame, size=2)
            glimpses.append(image[y:y+frame, x:x+frame])

            if k > 0:
                locations.append([y, x])

        instances.append(glimpses)

    images = np.reshape(instances, [batch_size, steps+1, frame, frame, 1])
    locations = np.reshape(locations, [batch_size, steps, 2])

    _, loss, rec = sess.run([model.optimizer, model.loss, model.reconstruction2d],
                            feed_dict={
                                model.observation: images[:, :-1, :, :, :],
                                model.target: images[:, 1:, :, :, :],
                                model.locations: locations
                            })

    if i % 1000 == 0:
        print(i, loss)

        inputs, targets, predictions = [], [], []

        for k in range(steps):
            inputs.append(images[0, k, :, :, :])
            targets.append(images[0, k+1, :, :, :])
            predictions.append(rec[k])

        inputs, targets, predictions = np.hstack(inputs), np.hstack(targets), np.hstack(predictions)

        cv2.imshow("images", np.vstack([inputs, targets, predictions]).astype(np.float32))
        cv2.waitKey(1)

    """
    if i % 1000 == 0:
        nx = ny = 20
        x_values = np.linspace(-3, 3, nx)
        y_values = np.linspace(-3, 3, ny)
        canvas = np.empty((frame * ny, frame * nx))
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = [xi, yi]
                x_mean = sess.run(model.reconstruction, feed_dict={model.z: [z_mu] * 5, model.locations: np.zeros([1, 5, 2])})
                canvas[(nx - i - 1) * frame:(nx - i) * frame, j * frame:(j + 1) * frame] = x_mean[0].reshape(frame, frame)

        cv2.imshow("gen", canvas)
        cv2.waitKey(1)
    """