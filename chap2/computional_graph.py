import tensorflow as tf
import numpy as np

sess = tf.Session()

x_vals = np.array([1., 3., 5., 7., 9.])
x_node = tf.placeholder(tf.float32)
y = tf.pow(x_node, 2)
for x_val in x_vals:
	feed_dict = {
		x_node:x_val
	}
	print sess.run(y, feed_dict)
