"""
visualize_mlflow

Created: 30.03.22, 01:10

Author: LDankert
"""

import pandas as pd
import ast
import numpy as np
from visual_audiolisation import plot_drum_matrix

mlflow_data = pd.read_csv("../data/mlflow_data.csv")
ebeats = mlflow_data["Epoch Beats"]
beats = mlflow_data["Generated Beat"]
ebeats = ebeats[:-1]
for beat in ebeats:
    #beat = beat.replace("tf.Tensor(\n", "")
    #beat = beat.replace(", shape=(1, 32, 9), dtype=float32)", "")
    #beat = beat.replace("\n", "")
    #beat = beat.replace("[[", "")
    #beat = beat.replace("]]", "")
    #print(np.fromstring(beat))
    print(ast.literal_eval(beat))




