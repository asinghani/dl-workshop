import numpy as np
import tensorflow as tf
import sys, glob, os
from PIL import Image
import random

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# x y w h

def create_tf_example(path, label_path):
    with tf.gfile.GFile(path, "rb") as tffile:
        image = tffile.read()

    width, height = Image.open(path).size[0:2]
    name, format = path.split("/")[-1].split(".")

    xmin = []
    xmax = []
    ymin = []
    ymax = []

    num_labels = 0

    with open(label_path) as file:
        labels = [line.strip() for line in file]

    for label in labels:
        label = label.split(" ")
        x = float(label[1])
        y = float(label[2])
        w = float(label[3])
        h = float(label[4])

        xmin.append(x - 0.5 * w)
        xmax.append(x + 0.5 * w)

        ymin.append(y - 0.5 * h)
        ymax.append(y + 0.5 * h)

        continue

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(str.encode(name + "." + format)),
        'image/source_id': bytes_feature(name),
        'image/encoded': bytes_feature(image),
        'image/format': bytes_feature(format),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(["robot"] * num_labels),
        'image/object/class/label': int64_list_feature([0] * num_labels),
    }))


random.seed(42)
train_test_split = 0.10 # 10% reserved for test/validation

img_dir = sys.argv[1]
label_dir = sys.argv[2]

images = glob.glob(os.path.join(img_dir, "*.jpg"))
random.shuffle(images)

images_test = images[0:int(train_test_split * len(images) + 1)]
images_train = images[int(train_test_split * len(images) + 1):]

# Make training set
writer = tf.python_io.TFRecordWriter("train.record")

for image in images_train:
    print("Train", image)
    image_name = image.split("/")[-1].split(".")[0]
    writer.write(create_tf_example(image, os.path.join(label_dir, image_name+".txt")).SerializeToString())

writer.close()

writer = tf.python_io.TFRecordWriter("test.record")

for image in images_test:
    image_name = image.split("/")[-1].split(".")[0]
    print("Test", image)
    writer.write(create_tf_example(image, os.path.join(label_dir, image_name+".txt")).SerializeToString())

writer.close()
