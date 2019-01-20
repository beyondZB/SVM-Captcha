import os
import struct
import numpy as np


def mnist_unpack(path, kind='train'):
    """load MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as labels_file:
        lmagic, lnum = struct.unpack(">II", labels_file.read(8))
        labels = np.fromfile(labels_file, dtype=np.uint8)

    with open(images_path, 'rb') as images_file:
        imagic, inum, irows, icols = struct.unpack(">IIII", images_file.read(16))
        vector_images = np.fromfile(images_file, dtype=np.uint8)
        images = vector_images.reshape(lnum, irows * icols)
    return images, labels


# import matplotlib.pyplot as plt
#
# images, labels = mnist_unpack('.\\')
# flg, ax = plt.subplots(
#     nrows=2,
#     ncols=5,
#     sharex=True,
#     sharey=True)
# ax = ax.flatten()
#
# for i in range(10):
#     img = images[i].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
#
# plt.tight_layout()
# plt.show()
