"""
This is the Main file for our Final project

TODO:
    Übergangslösung entfernen

Created: 29.03.22, 21:09

Author: LDankert

"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from util import BATCH_SIZE
from Generator import Generator
from Discriminator import Discriminator
from training_loop import training_loop

# Provide dataset
if not os.path.exists("data/cleaned_data.pkl"):
    print("No .pickle file found, dataset will be created:")
    try:
        os.system("python data_processing.py")
        dataset = pd.read_pickle("data/cleaned_data.pkl")
    except:
        print("Could not execute the data_processing.py file")
else:
    dataset = pd.read_pickle("data/cleaned_data.pkl")

#dataset = dataset.iloc[:20]
# solve the processing problem TODO: Übergangslösung
data = np.vstack(dataset["drum_matrices"])
data = tf.data.Dataset.from_tensor_slices(data)

# change datatype to float 32
data = data.map(lambda matrix: (tf.cast(matrix, tf.float32)))
# shuffle the datasets
data = data.shuffle(buffer_size=1000)
# batch the datasets
data = data.batch(BATCH_SIZE)
# prefetch the datasets
data = data.prefetch(20)

# Initialise generator and discriminator
generator = Generator()
discriminator = Discriminator()

training_loop(data, generator, discriminator, BATCH_SIZE, epochs=10)