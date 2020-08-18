import tensorflow as tf
import numpy as np

import os
import json

class SegNetModel () :
    def __init__ (self, conf_file="config.json") :
        with open (conf_file) as file :
            self.config = json.load (file)
        
        self.input_w = self.config["IMAGE_WIDTH"]
        self.input_h = self.config["IMAGE_HEIGHT"]
        self.input_c = self.config["INPUT_CHANNELS"]
        self.batch_size = self.config["BATCH_SIZE"]


    def build_model (self) :
        input_image = tf.keras.layers.Input (shape=[self.input_h, self.input_w, self.input_c], batch_size=self.batch_size)

        shape_0 = tf.shape(input_image)
        print (input_image.shape)
        x = tf.keras.layers.Conv2D (64, kernel_size=3, strides=1, padding='same') (input_image)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (64, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x, x0_indices = tf.nn.max_pool_with_argmax (x, 2, strides=2, padding='SAME')

        shape_1 = tf.shape(x)
        print (x.shape)
        x = tf.keras.layers.Conv2D (128, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (128, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x, x1_indices = tf.nn.max_pool_with_argmax (x, 2, strides=2, padding='SAME')

        shape_2 = tf.shape(x)
        print (x.shape)
        x = tf.keras.layers.Conv2D (256, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (256, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (256, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x, x2_indices = tf.nn.max_pool_with_argmax (x, 2, strides=2, padding='SAME')

        shape_3 = tf.shape(x)
        print (x.shape)
        x = tf.keras.layers.Conv2D (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x, x3_indices = tf.nn.max_pool_with_argmax (x, 2, strides=2, padding='SAME')

        shape_4 = tf.shape(x)
        print (x.shape)
        x = tf.keras.layers.Conv2D (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2D (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x, x4_indices = tf.nn.max_pool_with_argmax (x, 2, strides=2, padding='SAME') 

        x = up_sampling (x, x4_indices, shape_4, self.batch_size)

        x = tf.keras.layers.Conv2DTranspose (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x = up_sampling (x, x3_indices, shape_3, self.batch_size)

        x = tf.keras.layers.Conv2DTranspose (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (512, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (256, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x = up_sampling (x, x2_indices, shape_2, self.batch_size)

        x = tf.keras.layers.Conv2DTranspose (256, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (256, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (128, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x = up_sampling (x, x1_indices, shape_1, self.batch_size)

        x = tf.keras.layers.Conv2DTranspose (128, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (64, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x = up_sampling (x, x0_indices, shape_0, self.batch_size)

        x = tf.keras.layers.Conv2DTranspose (64, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)
        x = tf.keras.layers.Conv2DTranspose (self.num_classes, kernel_size=3, strides=1, padding='same') (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x = tf.nn.softmax (x)

        return tf.keras.Model (inputs=input_image, outputs=x)


def up_sampling(input, indices, output_shape, batch_size, name=None):
    pool_ = tf.reshape(input, [-1])
    batch_range = tf.reshape(tf.range(batch_size, dtype=indices.dtype), [tf.shape(input)[0], 1, 1, 1])
    b = tf.ones_like(indices) * batch_range
    b = tf.reshape(b, [-1, 1])
    ind_ = tf.reshape(indices, [-1, 1])
    ind_ = tf.concat([b, ind_], 1)
    ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
    ret = tf.reshape(ret, [tf.shape(input)[0], output_shape[1], output_shape[2], output_shape[3]])
    return ret

model = SegNetModel ()
gen = model.build_model ()

gen.summary ()
tf.keras.utils.plot_model (gen)