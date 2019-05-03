# python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os


def unpickle(data_file, encoding='ASCII'):
    with open(data_file, 'rb') as raw:
        raw_batch = pickle.load(raw, encoding=encoding)
    return raw_batch



def row_to_image_array(row):
    """
    Described in section "Dataset Layout" in https://www.cs.toronto.edu/~kriz/cifar.html
    """
    image = row.reshape((3, 32, 32))
    image[0] = np.rot90(image[0], 3)
    image[1] = np.rot90(image[1], 3)
    image[2] = np.rot90(image[2], 3)
    return image.T



def save_image_to_file(row, output_dir, label_name):
    image = row_to_image_array(row)
    plt.imsave(f'{output_dir}/{label_name}.jpg', image, origin='lower')



def save_images(raw_batch, names, output_dir):
    data = raw_batch[bytes('data', 'utf-8')]
    labels = raw_batch[bytes('labels', 'utf-8')]

    for idx, row in enumerate(data):
        save_image_to_file(row, output_dir, label_name=f'{names[labels[idx]]}_{idx}')
        if (idx+1) % 100 == 0:
            print('.', sep='', end='', flush=True)
        if (idx+1) % 10000 == 0:
            print()



def convert_and_save(cifar_data_dir, output_dir, mode='train'):
    os.makedirs(output_dir, exist_ok=True)
    label_names = unpickle(f'{cifar_data_dir}/batches.meta')['label_names']

    if mode == 'train':
        for i in range(5):
            raw_batch = unpickle(f'{cifar_data_dir}/data_batch_{i+1}', encoding='bytes')
            save_images(raw_batch, label_names, output_dir)
    else:
        raw_batch = unpickle(f'{cifar_data_dir}/test_batch', encoding='bytes')
        save_images(raw_batch, label_names, output_dir)



def build_args():
    parser = argparse.ArgumentParser(description="Convert raw cifar data in jpgs")
    parser.add_argument('-d', '--srcdir', default='./data/cifar-10-batches-py', help='cifar data dir')
    parser.add_argument('-m', '--mode', default='train', help='training or test data')
    parser.add_argument('-o', '--outdir', default='./data/images', help='output dir where jpgs will be saved')
    args = parser.parse_args()
    args.outdir = f'{args.outdir}/{args.mode}'

    return args



if __name__ == '__main__':
    args = build_args()
    print(f'Will convert cifar data in {args.srcdir} to jpgs into {args.outdir}')
    convert_and_save(args.srcdir, args.outdir, args.mode)
