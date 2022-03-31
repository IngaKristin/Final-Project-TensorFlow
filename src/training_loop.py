"""
training_loop

Created: 29.03.22, 21:06

Author: LDankert
"""
import datetime
import tensorflow as tf
import numpy as np

from tqdm import tqdm, trange
from util import NOTES_LENGTH, DRUM_CLASSES
from visual_audiolisation import plot_drum_matrix, play_drum_matrix

sequence_length = NOTES_LENGTH  # The length of the incoming drum matrices sequences

nb_notes = len(DRUM_CLASSES)  # Number of possible notes


#@tf.function
def training_loop(dataset, generator, discriminator, batch_size, epochs=10, visualize=False):
    """ This is the function for the training loop

    :param dataset: (tf.data,Dataset) The dataset for training
    :param generator: (tf.keras.Model) Generator Model
    :param discriminator: (tf.keras.Model) Discriminator Model
    :param batch_size: (int) Batch size (as used in preprocessing) to create batches of fake beates
    :param epochs: (int) Number of epochs to train, default=10
    :param visualize: (boolean) whether the drum matrices during the training process, default=True
    :return generator_loss (lst): Loss value of the generator for every epoch
    :return discriminator_loss (lst): Loss value of the discriminator for every epoch
    :return discriminator_acc (lst): Accuracy value for every epoch
    :return beats (lst): One drum matrix for each epoch
    """

    # accuracy metric
    acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)


    generator_loss, discriminator_loss, discriminator_acc, fake_beats = [], [], [], []

    for epoch in trange(epochs, leave=True, unit='epoch', desc=f"Training progress"):
        acc_aggregator = []
        gen_loss = 0
        disc_loss = 0

        for drum_matrix in tqdm(dataset):  # visualise epoch process
            #print(f"drum matrix{drum_matrix.shape}")
        #for drum_matrix in dataset:
            # random noise for generator
            noise = tf.random.normal(shape=(batch_size, sequence_length*nb_notes))

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_beat = generator(noise, training=True)

                # discriminator's output for both fake and real images
                real_output = discriminator(drum_matrix, training=True)
                #print(f"output{real_output.shape}")
                fake_output = discriminator(generated_beat, training=True)
                #print(f"fake_output : {fake_output.shape}")

                # loss functions for generator and discriminator, including the L2 regularization term
                gen_loss = generator.loss_function(tf.ones_like(fake_output), fake_output) + tf.reduce_sum(generator.losses)
                #print(f"gen_loss: {gen_loss}")
                disc_loss = discriminator.loss_function(tf.ones_like(real_output), real_output) + discriminator.loss_function(
                    tf.zeros_like(fake_output), fake_output) + tf.reduce_sum(discriminator.losses)

                labels = tf.concat((tf.ones_like(real_output), tf.zeros_like(fake_output)), axis=0)
                #print(f"labels: {labels.shape}")
                beats = tf.concat((real_output, fake_output), axis=0)
                #print(f"beats: {beats.shape}")
                acc_aggregator.append(acc(labels, beats))
                #print(f"acc: {acc_aggregator}")

            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator.optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator.optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # create 1 fake beat after each epoch
        fake_beat = generator(tf.random.normal(shape=(1, 288)), training=False)

        aggregated_acc = tf.reduce_mean(acc_aggregator)

        # matrices for mlflow
        generator_loss.append(gen_loss.numpy())
        discriminator_loss.append(disc_loss.numpy())
        discriminator_acc.append(aggregated_acc.numpy())
        fake_beats.append(fake_beat.numpy())

        # visualization of generator progress throughout the epochs
        if visualize:
            plot_drum_matrix(fake_beat.numpy())

    return generator_loss, discriminator_loss, discriminator_acc, fake_beats
