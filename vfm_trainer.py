from tensorflow.examples.tutorials.mnist import input_data
import cv2
import vfm_model
import tensorflow as tf
import numpy as np
import time

np.random.seed(1)
tf.set_random_seed(1)

flags = tf.app.flags
FLAGS = flags.FLAGS

def summary_float(step, name, value, summary_writer):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summary_writer.add_summary(summary, global_step=step)

def run():
    mnist = input_data.read_data_sets("data", False)

    frame = FLAGS.glimpse_size
    steps = FLAGS.time_steps
    batch_size = FLAGS.batch_size

    model = vfm_model.Model(frame, steps)

    saver = tf.train.Saver(max_to_keep=5)
    sess = tf.Session()
    writer = tf.summary.FileWriter(FLAGS.model_dir)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored %s" % ckpt.model_checkpoint_path)

    cv2.namedWindow("images", cv2.WINDOW_NORMAL)
    cv2.namedWindow("gen", cv2.WINDOW_NORMAL)
    cv2.namedWindow("extrapolate", cv2.WINDOW_NORMAL)

    while True:
        global_step = sess.run(model.global_step)

        def learn(images, labels, train):
            instances = []
            locations = []
            true_labels = []

            for image, label in zip(images, labels):
                glimpses = []
                image = np.reshape(image, [28, 28])

                for k in range(steps+1):
                    x, y = np.random.randint(low=0, high=28-frame, size=2)
                    glimpses.append(image[y:y+frame, x:x+frame])
                    locations.append([y, x])

                    if k > 0:
                        true_labels.append(label)

                instances.append(glimpses)

            images = np.reshape(instances, [batch_size, steps+1, frame, frame, 1])
            locations = np.reshape(locations, [batch_size, steps+1, 2])
            fd = {model.observation: images[:, :-1, :, :, :],
                  model.target: images[:, 1:, :, :, :],
                  model.locations: locations,
                  model.true_label: true_labels
                  }

            if train:
                global_step, _, loss, rec, class_p, img_error, dec, class_acc, avg_ce_dec = sess.run([model.global_step, model.optimizer, model.loss, model.prediction2d, model.class_test,
                                                                                                      model.BCE, model.avg_BCE_dec, model.class_acc, model.avg_ce_dec],
                                        feed_dict=fd)

                summary_float(global_step, "train loss", loss, writer)
                summary_float(global_step, "train image error", img_error, writer)
                summary_float(global_step, "train image error decrease", dec, writer)
                summary_float(global_step, "train z class accuracy", class_acc, writer)
                summary_float(global_step, "train cross entropy decrease", avg_ce_dec, writer)

                if global_step % 100 == 0:
                    print(global_step, loss)

                if global_step % 10000 == 0:
                    saver.save(sess, FLAGS.model_dir + "/model", global_step)
            else:
                global_step = sess.run(model.global_step)
                rec, class_p, img_error, dec, class_acc, avg_ce_dec = sess.run([model.prediction2d, model.class_test, model.BCE, model.avg_BCE_dec, model.class_acc, model.avg_ce_dec],
                                                                               feed_dict=fd)
                summary_float(global_step, "test image error", img_error, writer)
                summary_float(global_step, "test image error decrease", dec, writer)
                summary_float(global_step, "test z class accuracy", class_acc, writer)
                summary_float(global_step, "test cross entropy decrease", avg_ce_dec, writer)

                inputs, targets, predictions = [], [], []

                for k in range(steps):
                    inputs.append(images[0, k, :, :, :])
                    targets.append(images[0, k+1, :, :, :])
                    predictions.append(rec[k])
                    print("%i: %f" % (np.argmax(class_p[k]), np.amax(class_p[k])))

                inputs, targets, predictions = np.hstack(inputs), np.hstack(targets), np.hstack(predictions)

                cv2.imshow("images", np.vstack([inputs, targets, predictions]).astype(np.float32))
                cv2.waitKey(1)

                print("-"*10)

            return global_step

        train = False

        if train:
            images, labels = mnist.train.next_batch(batch_size)
            learn(images, labels, True)

            if global_step % 100 == 0:
                images, labels = mnist.test.next_batch(batch_size)
                learn(images, labels, False)

        if not train and global_step % 1000 == 0:
            time.sleep(5)
            images, labels = mnist.test.next_batch(batch_size)
            image = images[0]

            locations = [[0, 14], [14, 0], [14, 14]]
            image = np.reshape(image, [28, 28, 1])
            patches = []

            for x, y in locations:
                glimpses = []
                glimpses.append(image[0:frame, 0:frame])
                glimpses.append(image[y:y+frame, x:x+frame])
                images = np.reshape(glimpses, [batch_size, 2, frame, frame, 1])
                locations = np.reshape([[0, 0], [x, y]], [batch_size, 2, 2])
                fd = {model.observation: images[:, :-1, :, :, :],
                      model.target: images[:, 1:, :, :, :],
                      model.locations: locations
                      }
                rec = sess.run(
                    model.prediction2d,
                    feed_dict=fd)
                patches.append(rec[0])

            whole = np.empty([28, 28, 1])
            image[:frame, :frame] = 1.0 - image[:frame, :frame]
            whole[:14, :14] = image[:frame, :frame]
            whole[:14, 14:] = patches[0]
            whole[14:, :14] = patches[1]
            whole[14:, 14:] = patches[2]

            cv2.imshow("extrapolate", np.hstack([whole, image]))
            cv2.waitKey(1)

if __name__ == "__main__":
    flags.DEFINE_string("model_dir", "summaries/test00", "model dir")
    flags.DEFINE_integer("batch_size", 32, "batch size")
    flags.DEFINE_integer("time_steps", 10, "time steps")
    flags.DEFINE_integer("glimpse_size", 14, "glimpse_size")

    run()