from os.path import join
from os import listdir
import numpy as np
datapath = '/home/wenshan/datasets/quickdraw'

filecat = 'sketchrnn_cat.npz'

with np.load(join(datapath, filecat)) as cat_data:
	print type(cat_data), cat_data.keys()

