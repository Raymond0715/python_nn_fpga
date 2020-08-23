import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100

from pathlib import Path
import math
import numpy as np
import argparse
import importlib
import csv

parser = argparse.ArgumentParser(
        description = 'Specify key arguments for this project.')
parser.add_argument(
        '--model', default = 'resnet20', 
        help = 'Name of loaded model. Must be one of resnet20 and vgg16')
parser.add_argument(
        '--class_num', default = 10, type = int,
        help = 'Number of output class.')
parser.add_argument(
        '--dataset', default = 'cifar10', 
        help = 'Dataset.')
parser.add_argument(
        '--quantilize', default = 'full',
        help = 'Quantilization mode. Must be one of full, ste and ng. When full'
        'is set, quantilze_w and quantilize_x will be useless.')
parser.add_argument(
        '--quantilize_w', default = 32, type = int,
        help = 'Weights bits width for quantilize model ')
parser.add_argument(
        '--quantilize_x', default = 32, type = int,
        help = 'Activation bits width for quantilize model ')
parser.add_argument(
        '--weight_decay', default = 0.0005, type = float,
        help = 'Weight decay for regularizer l2.')
parser.add_argument(
        '--batch_size', default = 128, type = int,
        help = 'Numbers of images to process in a batch.')
parser.add_argument(
        '--num_epochs', default = 250, type = int, 
        help = 'Number of epochs to train. -1 for unlimited.')
parser.add_argument(
        '--learning_rate', default = 0.1, type = float,
        help = 'Initial learning rate used.')
parser.add_argument(
        '--log_dir', default = 'log_dir',
        help = 'Directory of log file. Always in `log` directory.')
parser.add_argument(
        '--log_file', default = 'log_file.csv',
        help = 'Name of log file.')
args = parser.parse_args()

# Param
lr_drop = 20
lr_decay = 1e-6
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs

def normalize(X_train,X_test):
    mean    = np.mean(X_train,axis=(0, 1, 2, 3))
    std     = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test  = (X_test-mean)/(std+1e-7)
    return X_train, X_test


def lr_scheduler(epoch):
    if epoch < 20:
        return 0.1
    elif epoch < 80:
        return 0.01
    elif epoch < 140:
        return 0.001
    elif epoch < 200:
        return 0.0001
    else:
        return 0.00001
    # return learning_rate * (0.5 ** (epoch // lr_drop))


class NGalpha(tf.keras.callbacks.Callback):
    def __init__(self):
        super(NGalpha, self).__init__()
        
    def on_epoch_begin(self, epoch, logs = None):
        self.model.alpha = \
                1.0 / (math.e - 1.0) * \
                (math.e ** (float(epoch) / self.model.num_epochs) - 1)

    def on_test_begin(self, logs = None):
        self.model.alpha = 1

    def on_epoch_end(self, epoch, logs = None):
        current_dir = Path.cwd()
        log_dir = current_dir / 'log' / args.log_dir
        log_dir.mkdir(parents = True, exist_ok = True)
        log_path = log_dir / args.log_file
        with open(log_path, 'a') as csvfile:
            csvwriter = csv.DictWriter(csvfile, logs.keys())
            if epoch == 0:
                csvwriter.writeheader()

            csvwriter.writerow(logs)


if __name__ == '__main__':
    if args.dataset == 'cifar10':
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif args.dataset == 'cifar100':
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')

    # Normalization
    x_train, x_test = normalize(x_train, x_test)

    y_train = tf.keras.utils.to_categorical(y_train, args.class_num)
    y_test  = tf.keras.utils.to_categorical(y_test, args.class_num)


    # model = cifar10vgg(False)
    # y_pred = model.predict(x_test)
    # m = tf.keras.metrics.Accuracy()
    # m.update_state(np.argmax(y_pred,1), np.argmax(y_test,1))
    # pred_accuracy = m.result().numpy()

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
            # featurewise_center = True,
            # featurewise_std_normalization = True,
            width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width) 
            height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height) 
            horizontal_flip = True,  # randomly flip images 
            vertical_flip = False)  # randomly flip images
    datagen.fit(x_train)

    #optimization details
    sgd = tf.keras.optimizers.SGD(
            lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model = importlib.import_module(
            '.' + args.model, 'models').model 

    # model.compile(
            # loss='categorical_crossentropy', optimizer=sgd,
            # metrics=['accuracy'], run_eagerly = True)
    model.compile(
            loss='categorical_crossentropy', optimizer=sgd,
            metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 20 epoches.
    historytemp = model.fit(
            datagen.flow(x_train, y_train, batch_size = batch_size), 
            steps_per_epoch=x_train.shape[0] // batch_size, 
            epochs=num_epochs, 
            validation_data=(x_test, y_test),
            callbacks=[
                reduce_lr,
                NGalpha()],
            verbose=2)
