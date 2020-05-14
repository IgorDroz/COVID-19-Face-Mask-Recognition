import os
import argparse
import numpy as np
import pandas as pd


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)

#####
# TODO - your prediction code here

# Example (A VERY BAD ONE):
y_pred = np.random.randint(2, size=len(files))
prediction_df = pd.DataFrame(zip(files, y_pred), columns=['id', 'label'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Example - Calculating F1 Score using sklrean.metrics.f1_score
# from sklearn.metrics import f1_score
# y_true = prediction_df['id'].apply(lambda x: int(x[7:8])).values
# f1 = f1_score(y_true, y_pred, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.

# print("F1 Score is: {:.2f}".format(f1))


