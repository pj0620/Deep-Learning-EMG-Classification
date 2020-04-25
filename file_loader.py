import numpy as np
from os import walk
from numpy import genfromtxt
import progressbar

def import_data():
    data_dir = "EMG_data_for_gestures-master"
    # num_of_test_subjects = 32
    num_of_test_subjects = 1
    data = [None]*num_of_test_subjects

    for i in progressbar.progressbar(range(num_of_test_subjects),suffix="| loading input data"):
        user_data_dir = data_dir + "/" + "{:02d}".format(i+1)
        files = []
        for (dirpath, dirnames, filenames) in walk(user_data_dir):
            files.extend(filenames)
            break
        data[i] = [None]*2
        for k,file in enumerate(files):
            data[i][k] = genfromtxt(user_data_dir + "/" + file)
            # remove row 0
            data[i][k] = np.delete(data[i][k], (0), axis=0)

    return data