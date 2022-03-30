"""
test

Created: 29.03.22, 09:22

Author: LDankert
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os

from Generator import Generator
from Discriminator import Discriminator
from visual_audiolisation import plot_drum_matrix, play_drum_matrix
from scipy.io import wavfile

dataset = pd.read_pickle("../data/cleaned_data.pkl")
#print(dataset["style"].value_counts())
#print(dataset.info())

#for matrix in dataset["drum_matrices"].iloc[4]:
#    plot_drum_matrix(matrix)

#audio_data = play_drum_matrix(dataset["drum_matrices"].iloc[5][6])

#wavfile.write("data/test.wav", 44100, audio_data)

#test_disc = Discriminator()
#test_data = tf.constant(dataset["drum_matrices"].iloc[5][6], dtype=tf.float32)
#test_data = tf.expand_dims(test_data,1)
#x = test_disc(test_data)
#test_disc.summary()

#test_gen = Generator()
#test_data = np.random.normal(0.0,1.0,size=288)
#test_data = tf.expand_dims(test_data,0)
#x = test_gen(test_data)
#test_gen.summary()
#plot_drum_matrix(x)
#audio_data = play_drum_matrix(x[0])
#wavfile.write("data/test.wav", 44100, audio_data)

dataset = np.vstack(dataset["drum_matrices"])
print(dataset.shape)

all_gen = []
generator = Generator()
fake_beat = generator(tf.random.normal(shape=(1, 288)), training=False)
all_gen.append(fake_beat)
print(all_gen)
#print(dataset["drum_matrices"].shape)