
import csv, os
from Data_Path import get_data_path


data_dir = get_data_path()

file_path = os.path.join(data_dir, 'fall11_urls.txt')

with open(file_path) as inf:
    reader = csv.reader(inf, delimiter=" ")
    second_col = list(zip(*reader))[1]
    # In Python2, you can omit the `list(...)` cast