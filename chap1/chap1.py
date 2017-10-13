import tensorflow as tf

row_dim = 5
col_dim = 5


# ----------------------------------
# tensor type

# Type 1: Fixed
# create zero filled tensor
zero_tf = tf.zeros([row_dim, col_dim])
# create one filled tensor
one_tf = tf.ones([row_dim, col_dim])
# create constant filled tensor
fill_tf = tf.fill([row_dim, col_dim], 23)
# create array in tensor
constant_tf = tf.constant([1,2,3])


# Type 2: similar shape
zero_tf_similar = tf.zeros_like(constant_tf)


# Type 3: Sequence 
linear_tf = tf.linspace(10.0, 12.0, 3) 
integer_sequence_tf = tf.range(start = 6, limit = 15, delta = 3)

# Type 4: Random
random_uniform_tf = tf.random_uniform([col_dim, row_dim], minval = 0, maxval = 1)

# Type 5: Shuffle
shuffle_tf = tf.random_shuffle(random_uniform_tf)

# Type 6: Random shuffle - Random crop
crop_size = [1,2]
crop_tf = tf.random_crop(random_uniform_tf, crop_size)

# ----------------------------------------------------
# tensor variable
# wrapping the tensor in the Variable() function

# 
my_var = tf.Variable(tf.zeros([row_dim, col_dim]))

#---------------------------------------------------
# Using Placeholders and Variables
# Placeholders and variables are key tools for using computational graphs in TensorFlow

print zero_tf