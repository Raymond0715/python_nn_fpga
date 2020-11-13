import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100

# from data import get_data
import data
from pathlib import Path
import math
import numpy as np
import argparse
import importlib
import csv
import datetime
import pdb
import sys


# Argument
parser = argparse.ArgumentParser(
        description = 'Specify key arguments for this project.')
parser.add_argument(
        '--model', default = 'resnet20', 
        help = 'Name of loaded model. Must be one of resnet20 and vgg16')
parser.add_argument(
        '--pretrain_path', default = None,
        help = 'Path of pretrain weight.')
parser.add_argument(
        '--class_num', default = 10, type = int,
        help = 'Number of output class.')
parser.add_argument(
        '--dataset', default = 'cifar10', 
        help = 'Dataset.')
parser.add_argument(
        '--mode', default = 'fit',
        help = 'Choose train function. Must be one of `fit` and `custom`.')
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
parser.add_argument(
        '--ckpt_dir', default = 'ckpt',
        help = 'Directory of ckpt. Always in `ckpt` directory.')
parser.add_argument(
        '--ckpt_file', default = 'model',
        help = 'Name of ckpt.')
parser.add_argument(
        '--device', default = '-1',
        help = 'Choose GPU.')
args = parser.parse_args()


def normalize(X_train,X_test):
    mean    = np.mean(X_train,axis=(0, 1, 2, 3))
    std     = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test  = (X_test-mean)/(std+1e-7)
    return X_train, X_test


# Param
lr_drop = 20
lr_decay = 1e-6
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
# Especially for imagenet
train_dataset_directory = \
        Path('/mnt') / 'ILSVRC2012' / 'ILSVRC2012_img_train'
val_dataset_directory = \
        Path('/mnt') / 'ILSVRC2012' / 'ILSVRC2012_img_val'

# def grad(model, inputs, targets):
    # with tf.GradientTape() as tape:
        # y_ = model(inputs, training = True)
        # loss_value = loss_object(y_true = targets, y_pred = y_)
    # return loss_value, tape.gradient(loss_value, model.trainable_variables)

def lr_scheduler(epoch):
    # # For SGD
    # if epoch < 80:
        # return 0.01
    # elif epoch < 140:
        # return 0.001
    # elif epoch < 200:
        # return 0.0001
    # else:
        # return 0.00001

    # For Adam
    # # Staircase 1
    # if epoch < 120:
        # return 0.0001
    # elif epoch < 200:
        # return 0.00001
    # else:
        # return 0.000001

    # Staircase 2
    if epoch < 80:
        return 0.0001
    elif epoch < 180:
        return 0.00001
    else:
        return 0.000001


class NGalpha(tf.keras.callbacks.Callback):
    def __init__(self):
        super(NGalpha, self).__init__()

    def on_epoch_begin(self, epoch, logs = None):
        if args.quantilize == 'ste':
            self.model.alpha.assign(0.0)
        elif args.quantilize == 'ng':
            self.model.alpha.assign(0.5)
        elif args.quantilize == 'full':
            pass
        else:
            print('[ERROR][main.py] Wrong quantilize param!!!')
            sys.exit()


    def on_test_begin(self, logs = None):
        # self.model.alpha[0] = 1.0
        # self.model.alpha.assign(1.0)
        self.model.alpha.assign(0.0)
        # pass
        # print('[INFO][main.py] test alpha:', self.model.alpha.)
        # self.model.alpha[0] = \
                # 1.0 / (math.e - 1.0) * \
                # (math.e ** (float(epoch) / self.model.num_epochs) - 1.0)

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
    # # Config GPU
    # if '-1' in args.device:
        # mirrored_strategy = tf.distribute.MirroredStrategy()
    # else:
        # device_index = args.device.split(',')
        # device_str = ['/gpu:' + s for s in device_index]
        # mirrored_strategy = tf.distribute.MirroredStrategy(devices = device_str)

    # with mirrored_strategy.scope():
        # optm = tf.keras.optimizers.SGD(
                # lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # model = importlib.import_module(
                # '.' + args.model, 'models').model

    # Define CNN architecture
    model = importlib.import_module('.' + args.model, 'models').model

    # Get dataset
    # train_dataset, val_dataset, steps_per_epoch, input_tensor_shape \
            # = data.GetData(args.dataset)
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

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # featurewise_center = True,
            # featurewise_std_normalization = True,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = True,
            vertical_flip = False)
    datagen.fit(x_train)

    # model architecture
    model = importlib.import_module(
            '.' + args.model, 'models').model

    # Load weights
    if args.pretrain_path != None:
        # model.build((None,) + x_train.shape[1:4])
        model.build((None,) + input_tensor_shape)
        pretrain_path = Path.cwd() / 'ckpt' / args.pretrain_path
        print('[INFO][main.py] Load weights from', pretrain_path)
        model.load_weights(str(pretrain_path))

    # Config model for train
    # optm = tf.keras.optimizers.SGD(
            # lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    optm = tf.keras.optimizers.Adam(lr=learning_rate)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        model.compile(
                loss='categorical_crossentropy', optimizer=optm,
                metrics=['accuracy', 'top_k_categorical_accuracy'])
        # model.compile(
                # loss='categorical_crossentropy', optimizer=optm,
                # metrics=['accuracy', 'top_k_categorical_accuracy'],
                # run_eagerly = True)
    elif args.dataset == 'imagenet':
        model.compile(
                loss='sparse_categorical_crossentropy', optimizer=optm,
                metrics=['accuracy', 'top_k_categorical_accuracy'])
    else:
        print('[ERROR][data.py] Wrong dataset!!!')
        quit()

    # adam = tf.keras.optimizers.Adam(lr = learning_rate)
    # model.compile(
            # loss='categorical_crossentropy', optimizer=adam,
            # metrics=['accuracy'])

    # # Learning rate call back
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    # model.build(tf.TensorShape([None, 32, 32, 3]))
    # model.build(tf.TensorShape([None, 224, 224, 3]))
    # for i, weight in enumerate(model.weights):
      # print('[DEBUG][inference4fpga.py] weight {} name: {}'.format(i, weight.name))
    # pdb.set_trace()

    # Training
    if args.mode == 'fit':
        historytemp = model.fit(
                datagen.flow(x_train, y_train, batch_size = batch_size),
                # train_dataset,
                steps_per_epoch=x_train.shape[0] // batch_size,
                # steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                validation_data=(x_test, y_test),
                # validation_data=val_dataset,
                callbacks=[
                    # reduce_lr,
                    NGalpha()],
                verbose=2)
    elif args.mode == 'custom':
        print('[INFO][main.py] Start training')
        train_loss_results = []
        train_accuracy_results = []
        for epoch in range(num_epochs):
            start_time = datetime.datetime.now()
            # Train
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
            for x, y in train_dataset:
                with tf.GradientTape() as tape:
                    y_ = model(x, training = True)
                    loss_value = loss_object(y_true = y, y_pred = y_)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optm.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_value)
                epoch_accuracy.update_state(y, y_)

            # End epoch 
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            # for i in range(len(val_dataset[0])):
            for (x, y) in val_dataset:
                logits = model(x, training = False)
                prediction = tf.argmax(
                        logits, axis = 1, output_type = tf.int32)
                test_accuracy(prediction, y)
                val_loss_value = loss_object(
                        y_true = y, 
                        y_pred = logits)

            end_time = datetime.datetime.now()
            delta_time = end_time - start_time
            print('Epoch {:03d}:'.format(epoch), end = '')
            print('Time {:.3f}, '.format(delta_time.total_seconds()), end = '')
            # epoch_loss_avg_num = epoch_loss_avg.result().numpy()
            # epoch_accuracy_num = epoch_accuracy.result().numpy()
            print('Train Loss: {:.3f}, Train Accuracy: {:.3f}%, '.format(
                epoch_loss_avg.result().numpy(), 
                epoch_accuracy.result().numpy()), end = '')
            # print('Train Loss: {:.3f}, Train Accuracy: {:.3f}%, '.format(
                # epoch_loss_avg_num, epoch_accuracy_num), end = '')
            pdb.set_trace()
            print('Val Loss: {:.3f}, Val Accuracy: {:.3f}%'.format(
                prediction, val_loss_value))
    else:
        print('[ERROR][main.py] Wrong args.mode!!!')
        quit()

    # Save model 
    current_dir = Path.cwd()
    ckpt_dir = current_dir / 'ckpt' / args.ckpt_dir
    ckpt_dir.mkdir(parents = True, exist_ok = True)
    ckpt_path = ckpt_dir / args.ckpt_file
    model.save_weights(str(ckpt_path))
