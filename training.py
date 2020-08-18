import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import cv2
import random

from Model import Model
from dataset import Dataset

from tqdm import tqdm
from PIL import Image

def load_mask (path) :
    input_mask = Image.open (MASKS_PATH + path.numpy().decode().split (".")[0] + ".png").convert ("P").resize ((IMG_HEIGHT, IMG_WIDTH))
    input_mask = np.array (input_mask)
    input_mask = np.where (input_mask>CLASSES, 0, input_mask)
    
    return input_mask

def load (image_file) :
    input_image = tf.io.read_file (IMAGES_PATH + image_file)
    input_image = tf.image.decode_jpeg (input_image)
    input_image = tf.cast (input_image, tf.float32)
    input_image = tf.image.resize (input_image, [IMG_HEIGHT, IMG_WIDTH])
    input_image = input_image / 255.0

    input_mask = tf.py_function (load_mask, [image_file], tf.uint8)

    return input_image, input_mask

def color_map (N=256, normalized=False) :
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((256, 3), dtype=dtype)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def calculate_loss (pred, gt) :
    gt = tf.one_hot(gt, depth=CLASSES)
    loss = tf.keras.losses.categorical_crossentropy (gt, pred, from_logits=True)
    return loss

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    return pred_mask

def generate_images(prediction, mask, image, epoch, num) :
    prediction = create_mask (prediction)
    mapping = color_map ()

    new_mask = np.zeros ((256, 256, 3))
    new_pred = np.zeros ((256, 256, 3))

    for i in range (256) :
        for j in range (256) :
            new_mask[i, j, :] = mapping[mask[0, i, j]]
            new_pred[i, j, :] = mapping[prediction[0, i, j]]

    plt.imsave ("Training/Epoch" + str (epoch) + "/Image" + str (num) + ".jpg", image[0])
    plt.imsave ("Training/Epoch" + str (epoch) + "/gt" + str (num) + ".jpg", new_mask)
    plt.imsave ("Training/Epoch" + str (epoch) + "/pred" + str (num) + ".jpg", new_pred)


def predict (dataset, epoch) :
    count = 0
    for n, (image, mask) in dataset.take(100).enumerate() :
        prediction = generator (image, training=True)
        generate_images (prediction, mask, image, epoch, n.numpy())

def train_step (image, mask, epoch) :
    with tf.GradientTape() as gen_tape :
        prediction = generator (image, training=True)
        loss = calculate_loss (prediction, mask)

    gen_gradients = gen_tape.gradient (loss, generator.trainable_variables)
    generator_optimizer.apply_gradients (zip (gen_gradients, generator.trainable_variables))

    return loss


def fit (train_dataset, epochs) :
    for epoch in range (START, epochs) :
        if not os.path.isdir ("Training/Epoch"+str(epoch)) :
            os.mkdir ("Training/Epoch"+str(epoch))

        print ("EPOCH : " + str (epoch))
        for image, mask in train_dataset :
            loss = train_step (image, mask, epoch)

        checkpoint.save (file_prefix=checkpoint_prefix)
        predict (train_dataset, epoch, "TRAIN")

IMAGES_PATH = "/home/gourav/Desktop/Image_Segmentation/PASCAL_VOC_2012/VOCdevkit/VOC2012/JPEGImages/"
MASKS_PATH = "/home/gourav/Desktop/Image_Segmentation/PASCAL_VOC_2012/VOCdevkit/VOC2012/SegmentationClass/"
IMAGE_FILE = "image_files_PASCAL.txt"
MASK_FILE = "mask_files_PASCAL.txt"
CLASSES = 21
BATCH_SIZE = 1
RESTORE_CHECKPOINT = True
BUFFER_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
EPOCHS = 100
START = 0

model = Model()
generator = model.build_generator (CLASSES)

train_file = open (IMAGE_FILE)
train_dataset = [line[:-1] for line in train_file.readlines()]
train_dataset = tf.data.Dataset.from_tensor_slices (train_dataset)
train_dataset = train_dataset.map (load)
train_dataset = train_dataset.shuffle (BUFFER_SIZE).batch (BATCH_SIZE)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay (initial_learning_rate=1e-4 * (0.96 ** START), decay_steps=50000, decay_rate=0.96, staircase=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

checkpoint_dir = './training_checkpoints'
if not os.path.isdir (checkpoint_dir) :
    os.mkdir (checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator=generator
)

if RESTORE_CHECKPOINT :
    checkpoint.restore (tf.train.latest_checkpoint (checkpoint_dir))

if not os.path.isdir ("Training") :
    os.mkdir ("Training")

fit (train_dataset, EPOCHS)