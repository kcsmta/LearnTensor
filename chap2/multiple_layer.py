import tensorflow as tf
import numpy as np
sess = tf.Session()

# size 3x5
my_array = np.array([[1., 3., 5., 7., 9.],
					[2., 4., 6. ,8., 10.],
					[-6., -3., 0., 3., 6.]])
# print my_array

x_vals = np.array([my_array, my_array+1])
# print x_vals

x_node =  tf.placeholder(tf.float32, shape = (3,5))

# size 5x1
m1 = tf.constant([[1.], [3.], [-1], [-2], [3]])

# size 1x1
m2 = tf.constant([[2.0]])

# size 1x1
a1 = tf.constant([[10.]])

# size 3x5 * 5x1 = 3x1
prod1 = tf.matmul(x_node, m1)

#size 3x1 * 1x1 = 3x1
prod2 = tf.matmul(prod1, m2)

add1 = tf.add(prod2, a1)

for x_val in x_vals:
	print sess.run(add1, feed_dict = {x_node:x_val})