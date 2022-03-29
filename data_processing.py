"""
This module for processes the data. The data for training only should contain the
beat "beat_type" not fill, because the fills are way too short. As well all unnecessary
information are dropped.

Created: 28.03.22, 14:51

Author: LDankert
"""

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

from util import CHOOSEN_GENRE
from preprocessing_functions import duplicate_multiple_styles, get_pianomatrices_of_drums

import os.path
import pandas as pd


# Download datasets
if not os.path.exists("data/groove"):
    print("Download started")
    http_response = urlopen("https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip")
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path="data")
    print("Download finished")
else:
    print("Load file from local")

dataset = pd.read_csv("data/groove/info.csv")

dataset_cleaned = pd.DataFrame()

# duplicates all songs that fit into multiple styles
for _, row in dataset.iterrows():
    dataset_cleaned = dataset_cleaned.append(duplicate_multiple_styles(row))

# remove all midi files that are not long enough
dataset_cleaned = dataset_cleaned[dataset_cleaned.beat_type != "fill"]

# just keep the filepath and style
dataset_cleaned = dataset_cleaned[["style", "midi_filename"]]

# just keep the styles with the most songs
print(f"Uses 5 most styles:{CHOOSEN_GENRE}")
dataset_cleaned = dataset_cleaned[dataset_cleaned["style"].isin(CHOOSEN_GENRE)]

# add correct file path to filename
dataset_cleaned["midi_filename"] = "data/groove/" + dataset_cleaned["midi_filename"]

# translates the midi file into a drum matrix
print("Start translating midi into drum matrix:")
dataset_cleaned["drum_matrices"] = [get_pianomatrices_of_drums(midi_file) for midi_file
                                    in dataset_cleaned["midi_filename"]]
print("Translation finished!")

# save cleaned_data as pickle file for later use
dataset_cleaned.to_pickle("data/cleaned_data.pkl")
