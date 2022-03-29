"""
This is the class of the Discriminator, for the GAN model

Created: 29.03.22, 19:00

Author: LDankert
"""

import tensorflow as tf
from tensorflow.keras import layers

from util import NOTES_LENGTH, DRUM_CLASSES


sequence_length = NOTES_LENGTH  # The length of the incoming drum matrices sequences

nb_notes = len(DRUM_CLASSES)  # Number of possible notes

droprate = 0


class Discriminator(tf.keras.Model):
    """Discriminator part of the GAN model"""

    def __init__(self, optimizer=tf.keras.optimizers.RMSprop()):
        """ Initializer

        :param optimizer: (tf.keras.optimizers.Optimizer) optimizer to use in training,
                        default RMSprop.
        """
        super(Discriminator, self).__init__()

        self.optimizer = optimizer

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.input_layer = layers.Input(shape=(sequence_length, nb_notes))
        self.all_layers = [
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, activation="tanh",
                                             dropout=droprate, recurrent_dropout=droprate)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, activation="tanh",
                                             dropout=droprate, recurrent_dropout=droprate)),
            layers.Dense(1, activation="sigmoid")
        ]

        self.out = self.call(self.input_layer, training=True)

    def call(self, x, training=False):
        """ Call function of the model. Propagates input through the layers

        :param x: (tensor) input to the model
        :param training: (boolean) whether to use training or inference mode, default: False (inference)
        :return: (tensor) tensor x after running through the model
        """
        for single_layers in self.all_layers:
            try:
                x = single_layers(x, training=training)
            except:
                x = single_layers(x)
        return x
