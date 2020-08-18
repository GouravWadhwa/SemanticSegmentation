import tensorflow as tf

import numpy as np
import os

from PIL import Image

class Dataset () :
    def __init__ (self, image_path, masks_path, image_file, mask_file, classes, batch_size=1) :
        self.image_path = image_path
        self.masks_path = masks_path

        self.image_file = [line[:-1] for line in open (image_file).readlines()]
        self.masks_file = [line[:-1] for line in open (mask_file).readlines()]

        self.mapping = self.color_map ()
        self.classes = classes
        self.batch_size = batch_size

    def color_map (self, N=256, normalized=False) :
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

    def get_size (self) :
        return len (self.image_file)

    def get_batch (self, i) :
        images = []
        masks = []
        if int (self.get_size () / self.batch_size) > i :
            for a in range (self.batch_size) :
                image, mask = self.get_item (self.batch_size * i + a)
                images.append (image)
                masks.append (mask)
        elif int (self.get_size () / self.batch_size) == i  :
            for a in range (self.get_size () - i * self.batch_size) :
                image, mask = self.get_item (self.batch_size * i + a)
                images.append (image)
                masks.append (mask)
        else :
            return None, None

        return np.array (images), np.array (masks)

    def get_item (self, i) :
        image = Image.open (os.path.join (self.image_path, self.image_file[i])).convert ("RGB")
        mask = Image.open (os.path.join (self.masks_path, self.masks_file[i])).convert ("P")

        image = np.array (image.resize ((256, 256)))
        mask = np.array (mask.resize ((256, 256)))

        mask = np.where (mask > self.classes, 0, mask)

        return image, mask

    def __call__ (self) :
        if self.get_size  () % self.batch_size == 0 :
            count = int (self.get_size () / self.batch_size)
        else :
            count = int (self.get_size () / self.batch_size) + 1
        images = []
        masks = []
        for i in range (count) :
            image, mask = self.get_batch (i)
            images.append (image)
            masks.append (mask)

        return [(images[i], masks[i]) for i in range(0, len(images))] 
