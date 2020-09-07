import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100

import pathlib
import numpy as np
import os
import pdb
from main import (
        args, train_dataset_directory, val_dataset_directory, 
)


AUTOTUNE = tf.data.experimental.AUTOTUNE


# Especially for cifar
def Normalize(X_train,X_test):
    mean    = np.mean(X_train,axis=(0, 1, 2, 3))
    std     = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test  = (X_test-mean)/(std+1e-7)
    return X_train, X_test


# Especially for cifar
def DataAugment(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    # Normalization
    x_train, x_test = Normalize(x_train, x_test)

    y_train = tf.keras.utils.to_categorical(y_train, args.class_num)
    y_test  = tf.keras.utils.to_categorical(y_test, args.class_num)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset   = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def Preprocess(img, label):
        img = tf.image.random_crop(img, [29, 29, 3])
        img = tf.image.resize(img, [32, 32])
        img = tf.image.random_flip_left_right(img)
        return img, label

    train_dataset = train_dataset.map(
            Preprocess, num_parallel_calls = AUTOTUNE)
    val_dataset   = val_dataset.map(
            Preprocess, num_parallel_calls = AUTOTUNE)

    return ConfigureForPerformance(train_dataset, args.batch_size), \
            ConfigureForPerformance(val_dataset, args.batch_size), \
            x_train.shape[0] // args.batch_size, \
            x_train.shape[1:4]

    # datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
            # width_shift_range = 0.1, 
            # height_shift_range = 0.1, 
            # horizontal_flip = True, 
            # vertical_flip = False) 
    # datagen.fit(x_train)
    # return datagen.flow(x_train, y_train, batch_size = args.batch_size), \
            # (x_test, y_test), \
            # x_train.shape[0] // args.batch_size, \
            # x_train.shape[1:4]


# Especially for imagenet
def GetImgLabelTrain(log, dataset_directory):
    log_list = tf.strings.split(log)
    train_dataset_directory_tf = tf.constant(str(dataset_directory))
    file_path = tf.strings.join(
            [train_dataset_directory_tf, log_list[0]], separator = os.path.sep)
    # img = tf.io.read_file(str(file_path))
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    # Normalization
    img = tf.image.resize(img, [256, 256])
    # img = tf.div(tf.add(tf.to_float(img), -127), 128)
    img = tf.math.divide(tf.math.add(tf.cast(img, tf.float32), -127), 128)

    img = tf.image.random_crop(img, [221, 221, 3])
    img = tf.image.resize(img, [224, 224])
    img = tf.image.random_flip_left_right(img)
    return img, tf.strings.to_number(log_list[1], tf.int32)


# Especially for imagenet
def GetImgLabelVal(log, dataset_directory):
    log_list = tf.strings.split(log)
    train_dataset_directory_tf = tf.constant(str(dataset_directory))
    file_path = tf.strings.join(
            [train_dataset_directory_tf, log_list[0]], separator = os.path.sep)
    # img = tf.io.read_file(str(file_path))
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    # Normalization
    img = tf.image.resize(img, [256, 256])
    # img = tf.math.div(tf.math.add(tf.to_float(img), -127), 128)
    img = tf.math.divide(tf.math.add(tf.cast(img, tf.float32), -127), 128)

    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    return img, tf.strings.to_number(log_list[1], tf.int32)


# Especially for imagenet
def ConfigureForPerformance(ds, batch_size):
    ds = ds.shuffle(buffer_size = 1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    return ds


def GetData(dataset):
    '''
    Args:
        dataset: Name of dataset. String.
    Returns:
        Train dataset. tf.data.dataset or tuple.
        Validation dataset. tf.data.dataset or tuple.
        Steps per epoch.
        Input tensor shape.
    '''
    if dataset == 'cifar10':
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return DataAugment(x_train, y_train, x_test, y_test)
    elif dataset == 'cifar100':
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        return DataAugment(x_train, y_train, x_test, y_test)
    elif dataset == 'imagenet':
        # Load text path
        train_label_path = pathlib.Path('.') / 'train.txt'
        val_label_path   = pathlib.Path('.') / 'val.txt'

        # Load text dataset
        train_txt_dataset = tf.data.TextLineDataset(str(train_label_path))
        val_txt_dataset   = tf.data.TextLineDataset(str(val_label_path))

        # Convert text dataset to image dataset
        train_dataset = train_txt_dataset.map(
                lambda x: GetImgLabelTrain(x, train_dataset_directory),
                num_parallel_calls = AUTOTUNE)
        val_dataset   = val_txt_dataset.map(
                lambda x: GetImgLabelVal(x, val_dataset_directory),
                num_parallel_calls = AUTOTUNE)
        batch_size = args.batch_size
        return ConfigureForPerformance(train_dataset, batch_size), \
                ConfigureForPerformance(val_dataset, batch_size), \
                None, (224, 224, 3)
