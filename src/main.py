"""
This is the Main file for our Final project

Created: 29.03.22, 21:09

Author: LDankert

"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from mlflow import log_param, log_metric, set_tracking_uri
from util import BATCH_SIZE
from Generator import Generator
from Discriminator import Discriminator
from training_loop import training_loop
from preprocessing_functions import data_processing
from visual_audiolisation import plot_drum_matrix

# Add arguments for running on grid
parser = argparse.ArgumentParser(description="Main training file")
parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs", default=10)
parser.add_argument("-s", "--save_model", help="Path to save the trained model", default=None)
parser.add_argument("-v", "--visualize", action="store_true", help="visualize drum matrices between epochs", default=True)
parser.add_argument("--RMSProp", type=float, help="Use Optimizer RMSProp learning rate x for training", default=None)
parser.add_argument("--SGD", type=float, help="Use Optimizer Stochastic gradient descent learning rate x for training", default=None)
parser.add_argument("--adam", type=float, help="Use Adam descent learning rate x for training", default=None)
parser.add_argument("--log_folder", help="Where to log the mlflow results", default="../data/mlflow")
args = parser.parse_args()

# setting log folder for mlflow grid data
set_tracking_uri(args.log_folder)

# Provide dataset
if not os.path.exists("../data/cleaned_data.pkl"):
    print("No .pickle file found, dataset will be created:")
    try:
        os.system("python data_processing.py")
        dataset = pd.read_pickle("../data/cleaned_data.pkl")
    except:
        print("Could not execute the data_processing.py file")
else:
    dataset = pd.read_pickle("../data/cleaned_data.pkl")

# makes every drum matrix to a single training point
data = np.vstack(dataset["drum_matrices"].iloc[:10])
data = data_processing(data)

# Choose Optimizer
if args.RMSProp is not None:
    print(f"   Training with RMSProp and LR = {args.RMSProp}")
    log_param("RMSProp", args.RMSProp)  # mlflow logs
    optimizer = tf.keras.optimizers.RMSprop(lr=args.RMSProp)

elif args.SGD is not None:
    print(f"   Training with SGD and LR = {args.SGD}")
    log_param("SGD", args.SGD)  # mlflow logs
    optimizer = tf.keras.optimizers.SGD(lr=args.SGD)

elif args.adam is not None:
    print(f"   Training with SGD and LR = {args.adam}")
    log_param("Adam", args.adam)  # mlflow logs
    optimizer = tf.keras.optimizers.Adam(lr=args.adam)

# Initialise generator and discriminator
generator = Generator(optimizer=optimizer)
discriminator = Discriminator(optimizer=optimizer)

# The final training process
gen_loss, disc_loss, disc_acc, beats = training_loop(data, generator, discriminator,
                                                     BATCH_SIZE, epochs=args.epochs,
                                                     visualize=args.visualize)
# Save the results in mlflow
log_param("Generator Loss", gen_loss)
log_param("Discriminator Loss", disc_loss)
log_param("Discriminator Accuracy", disc_acc)
log_param("Epoch Beats", beats)

# Save one drum matrix for mlflow
fake_beat = generator(tf.random.normal(shape=(1, 288)), training=False)
log_param("Generated Beat", fake_beat)


# Second training loop for rock genre
rock_data = dataset[dataset["style"] == "rock"]
print(rock_data.head)
#generator = generator # TODO: rdundant
#rock_discriminator = Discriminator()

#roch_training_loop(rockdata, genredata, generator, rock_discriminator, BATCH_SIZE, epochs=args.epochs)
