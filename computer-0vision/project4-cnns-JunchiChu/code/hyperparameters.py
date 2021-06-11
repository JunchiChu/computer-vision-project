"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

# Resize image size for task 1. Task 2 must have an image size of 224,
# so that is hard coded in elsewhere.
img_size = 224

# The number of image scene classes. Don't change this.
category_num = 15

# Sample size for calculating the mean and standard deviation of the
# training data. This many images will be randomly seleted to be read
# into memory temporarily.
preprocess_sample_size = 400

# Training parameters

# num_epochs is the number of epochs. If you experiment with more
# complex networks you might need to increase this. Likewise if you add
# regularization that slows training.
num_epochs = 50

# batch_size defines the number of training examples per batch:
# You don't need to modify this.
batch_size = 10

# learning_rate is a critical parameter that can dramatically affect
# whether training succeeds or fails. For most of the experiments in this
# project the default learning rate is safe.
learning_rate = 1e-4

# Momentum on the gradient (if you use a momentum-based optimizer)
momentum = 0.01
