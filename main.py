from file_loader import import_data
from feature_extraction import extract_features
import numpy as np

# load input data
all_data = import_data()
input_data = all_data[0][0]

# split into training and testing data
perc_train = 0.8
raw_train_data = input_data[:int(perc_train*input_data.shape[0]),:]
raw_test_data = input_data[int(perc_train*input_data.shape[0]):,:]

# extract features
window_size = 3000
train_data = extract_features(raw_train_data,window_size)