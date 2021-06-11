"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same", activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        # ============================================================================

        # TODO: Make all layers in self.vgg16 non-trainable. This will freeze the
        #       pretrained VGG16 weights into place so that only the classificaiton
        #       head is trained.


        # TODO: Write a classification head for our 15-scene classification task.
        #       Hint: The layers Flatten and Dense are essential here.

        self.head = []

        # ============================================================================

    def call(self, img):
        """ Passes the image through the network. """

        for layer in self.vgg16:
            img = layer(img)

        for layer in self.head:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
