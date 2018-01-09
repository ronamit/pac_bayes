import os
from os.path import expanduser

def get_data_path():
    # The path of the directory in which raw data is saved:
    data_path = os.path.join(expanduser("~"), 'ML_data_sets')
    return data_path

