import tensorflow as tf
from PIL import Image
import numpy as numpy
import pickle
import os
import random
import glob

class KTSDataset():
    def __init__(self, data_address="kts", random_seed=-1):
        self.database_address = data_address
        self.random_seed = random_seed
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
    def _get_parse_function(self):
        def parse_function(img_path):
            image = tf.image.decode_jpeg(tf.io.read_file(img_path))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.
        return parse_function

    def make_dataset(self, mode):
        images = []
        tags = []   
        self.num_data = 0
        with open("kts\\train.pickle", "rb") as fr:
            dataset = pickle.load(fr)     
        for data in dataset:
            if data["hashtag"] != []:
                images.append(data["img_name"])
                tags.append(random.choice(data["hashtag"]))
                self.num_data += 1
        return images, tags

    def get_dataset(self,
                    folders = "train",
                    batch_size = 10,
                    reshuffle_each_iteration=True):
        imgs, tags = self.make_dataset(folders)
        tag_dataset = tf.data.Dataset.from_tensor_slices(tags)
        img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
        img_dataset = img_dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)        

        dataset = tf.data.Dataset.zip((img_dataset, tag_dataset))
        dataset = dataset.shuffle(buffer_size=self.num_data, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        return dataset
    
database = KTSDataset("kts")

dataset = database.get_dataset("train", batch_size = 10)

for img, tag in dataset:
    print(img.shape)
    print(tag[0].numpy().decode())