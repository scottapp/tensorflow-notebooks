import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses


if __name__ == '__main__':

    dataset_dir = os.path.join(os.path.dirname('.'), 'aclImdb')
    if not os.path.isdir(dataset_dir):
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        dataset = tf.keras.utils.get_file("aclImdb_v1",
                                          url,
                                          untar=True,
                                          cache_dir='.',
                                          cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    assert os.path.isdir(dataset_dir), 'error dataset dir'

    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.isdir(remove_dir):
        shutil.rmtree(remove_dir)
