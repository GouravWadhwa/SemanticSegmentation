import tensorflow as tf

class Model () :
    def double_conv (self, x, filters, kernel_size) :
        initializer = tf.random_normal_initializer (0, 0.02)

        x = tf.keras.layers.Conv2D (filters=filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer) (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        x = tf.keras.layers.Conv2D (filters=filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer) (x)
        x = tf.keras.layers.BatchNormalization () (x)
        x = tf.keras.layers.ReLU () (x)

        return x

    def build_generator (self, out_channels) :
        input_image = tf.keras.layers.Input (shape=[None, None, 3])

        skip_1 = self.double_conv (input_image, 64, 3)
        x = tf.keras.layers.MaxPool2D () (skip_1)
        skip_2 = self.double_conv (x, 128, 3)
        x = tf.keras.layers.MaxPool2D () (skip_2)
        skip_3 = self.double_conv (x, 256, 3)
        x = tf.keras.layers.MaxPool2D () (skip_3)
        skip_4 = self.double_conv (x, 512, 3)
        x = tf.keras.layers.MaxPool2D () (skip_4)
        x = self.double_conv (x, 512, 3)
        x = tf.keras.layers.Conv2DTranspose (filters=512, kernel_size=3, strides=2, padding='same') (x)
        x = self.double_conv (x, 256, 3)
        x = tf.keras.layers.Concatenate () ([x, skip_4])
        x = tf.keras.layers.Conv2DTranspose (filters=256, kernel_size=3, strides=2, padding='same') (x)
        x = self.double_conv (x, 128, 3)
        x = tf.keras.layers.Concatenate () ([x, skip_3])
        x = tf.keras.layers.Conv2DTranspose (filters=128, kernel_size=3, strides=2, padding='same') (x)
        x = self.double_conv (x,64, 3)
        x = tf.keras.layers.Concatenate () ([x, skip_2])
        x = tf.keras.layers.Conv2DTranspose (filters=64, kernel_size=3, strides=2, padding='same') (x)
        x = self.double_conv (x, 64, 3)
        x = tf.keras.layers.Concatenate () ([x, skip_1])
        x = tf.keras.layers.Conv2D (filters=out_channels, kernel_size=1, strides=1, padding='same') (x)
        
        return tf.keras.Model (inputs=input_image, outputs=x)


model = Model ()
gen = model.build_generator (2)
gen.summary ()
tf.keras.utils.plot_model (gen)