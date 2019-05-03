# python3

import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_batch(path):
    with open(path, 'rb') as raw:
        raw_batch = pickle.load(raw, encoding='bytes')
    return raw_batch


def row_to_image_array(row):
    """
    Described in section "Dataset Layout" in https://www.cs.toronto.edu/~kriz/cifar.html
    """
    return row.reshape((3, 32, 32))


def save_image_to_file(row, dir, label_name):
    image = row_to_image_array(row)
    plt.imsave(f'{dir}/{label_name}.jpg', image)


def save_images(raw_batch, names, dir):
    data = raw_batch['data']
    labels = raw_batch['labels']

    for idx, row in enumerate(data):
        save_image_to_file(row, dir, label_name=f'{names[labels[idx]]}_{idx}')
