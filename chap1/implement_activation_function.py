import tensorflow as tf
sess = tf.Session()

# Relu function
print sess.run(tf.nn.relu6([-1.0, 4.0, -9.0, 1.0, 2.0]))
# sigmoid function
print sess.run(tf.nn.sigmoid([-1.0, 4.0, -9.0, 1.0, 2.0]))
# hyper tangent function
print sess.run(tf.nn.tanh([-1.0, 0, 1.0]))
# softsign function
print sess.run(tf.nn.softsign([-1.0, 0, 1.0, 4.0, -4.0]))
# softplus function
print sess.run(tf.nn.softplus([-1.0, 0, 1.0]))
# Enponential Linear Unit
print sess.run(tf.nn.elu([-1., 0., -1.]))

