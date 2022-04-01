"""
visualize_mlflow

Created: 30.03.22, 01:10

Author: LDankert
"""

import pandas as pd

mlflow_data = pd.read_csv("../data/mlflow/grid_search_data.csv")
print(mlflow_data)