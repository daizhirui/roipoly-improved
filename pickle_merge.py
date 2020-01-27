import pickle
import numpy as np
from tqdm import tqdm
import sys


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python3 pickle_merge.py [pickle files]')
        exit(0)
    pickle_files = sys.argv[1:]
    new_dict = {}
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            a = pickle.load(f)
        for key, val in tqdm(a[0].items()):
            new_dict[key] = np.array(val, dtype=np.uint8)
    with open('mask.pickle', 'wb') as f:
        pickle.dump((new_dict, a[1]), f)
