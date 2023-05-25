import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import ImageFilter
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255


def initialize_weights_biases(dim):
    w = np.zeros(shape=(dim,1))
    b=0
    return w,b


w, b = initialize_weights_biases(10)
print(w)
print(b)