"""
training_loop

Created: 29.03.22, 21:06

Author: LDankert
"""
import datetime
import tensorflow as tf

from tqdm import tqdm, trange
from util import NOTES_LENGTH, DRUM_CLASSES
from visual_audiolisation import plot_drum_matrix, play_drum_matrix

sequence_length = NOTES_LENGTH  # The length of the incoming drum matrices sequences

nb_notes = len(DRUM_CLASSES)  # Number of possible notes


#@tf.function
def training_loop(dataset, generator, discriminator, batch_size, epochs=10, visualize=True):
    """ This is the function for the training loop

    :param dataset: (tf.data,Dataset) The dataset for training
    :param generator: (tf.keras.Model) Generator Model
    :param discriminator: (tf.keras.Model) Discriminator Model
    :param batch_size: (int) Batch size (as used in preprocessing) to create batches of fake beates
    :param epochs: (int) Number of epochs to train, default=10
    :param visualize: (boolean) whether the drum matrices during the training process, default=True
    :return: None
    """
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_path = f"./logs/gan/train" + time
    train_summary_writer = tf.summary.create_file_writer(train_path)

    # accuracy metric
    acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

    for epoch in trange(epochs, leave=True, unit='epoch', desc=f"Training progress"):
        acc_aggregator = []

        for drum_matrix in tqdm(dataset):  # visualise epoch process
        #for drum_matrix in dataset:
            # random noise for generator
            noise = tf.random.normal(shape=(batch_size, sequence_length*nb_notes))

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_beat = generator(noise, training=True)

                # discriminator's output for both fake and real images
                real_output = discriminator(drum_matrix, training=True)
                fake_output = discriminator(generated_beat, training=True)

                # loss functions for generator and discriminator, including the L2 regularization term
                gen_loss = generator.loss_function(tf.ones_like(fake_output), fake_output) + tf.reduce_sum(generator.losses)
                disc_loss = discriminator.loss_function(tf.ones_like(real_output), real_output) + discriminator.loss_function(
                    tf.zeros_like(fake_output), fake_output) + tf.reduce_sum(discriminator.losses)

                # computation of accuracy
                labels = tf.concat((tf.ones_like(real_output), tf.zeros_like(fake_output)), axis=0)
                images = tf.concat((real_output, fake_output), axis=0)
                acc_aggregator.append(acc(labels, images))

            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator.optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator.optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

            # create 1 fake beat after each epoch
        fake_beat = generator(tf.random.normal(shape=(1, 288)), training=False)

        # log metrics for tensorboard
        aggregated_acc = tf.reduce_mean(acc_aggregator)
        with train_summary_writer.as_default():
            tf.summary.scalar(name="Generator loss", data=gen_loss, step=epoch)
            tf.summary.scalar(name="Discriminator loss", data=disc_loss, step=epoch)
            tf.summary.scalar(name="Discriminator acc", data=aggregated_acc, step=epoch)
            tf.summary.image(name="Generated beats", data=tf.expand_dims(fake_beat, -1), step=epoch)

        # visualization of generator progress throughout the epochs
        if visualize:
            plot_drum_matrix(fake_beat.numpy())
