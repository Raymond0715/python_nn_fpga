import tensorflow as tf
import scipy.io
# import matplotlib.pyplot as plt
import pathlib
import os


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    class_name = parts[-2]
    return class_name


# Param
data_dir = '/mnt'


data_dir = pathlib.Path(data_dir)
train_ds = tf.data.Dataset.list_files(
        str(data_dir / 'ILSVRC2012/ILSVRC2012_img_train/*/*.JPEG'), 
        shuffle = False)
val_ds = tf.data.Dataset.list_files(
        str(data_dir / 'ILSVRC2012/ILSVRC2012_img_val/*'), 
        shuffle = False)
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

for f in train_ds.take(5):
    print(get_label(f.numpy()))

# plt.figure()
# for f in train_ds.take(1):
    # img = tf.io.read_file(f.numpy())
    # img = tf.image.decode_jpeg(img, channels = 3)
    # plt.imshow(img.numpy())

mat = scipy.io.loadmat('meta.mat')['synsets']
imagenet_id_dict = {entry[0][1][0]:entry[0][0][0][0] for entry in mat}
print(len(imagenet_id_dict))
