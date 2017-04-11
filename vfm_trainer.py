from tensorflow.examples.tutorials.mnist import input_data
#import cv2
import vfm_model

mnist = input_data.read_data_sets("data", False)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

images, labels = mnist.train.next_batch(8)

#cv2.imshow("image", images[0])

model = vfm_model.Model()