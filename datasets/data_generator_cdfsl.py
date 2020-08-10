import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os

from glob import glob
from typing import Tuple, List
from PIL import Image
from collections import defaultdict
from abc import ABC, abstractmethod
from datetime import datetime


class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.raw_database_address = raw_database_address
        self.database_address = database_address

        self.input_shape = self.get_input_shape()
        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()

        self.train_folders = self.convert_to_dict(self.train_folders)
        self.val_folders = self.convert_to_dict(self.val_folders)
        self.test_folders = self.convert_to_dict(self.test_folders)

        if random_seed != -1:
            random.seed(None)

    @abstractmethod
    def get_train_val_test_folders(self):
        pass

    @abstractmethod
    def _get_parse_function(self):
        pass

    @abstractmethod
    def preview_image(self, image_path):
        pass

    @abstractmethod
    def get_input_shape(self):
        pass

    def convert_to_dict(self, folders):
        if type(folders) == list:
            classes = dict()
            for folder in folders:
                instances = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
                classes[folder] = instances

            folders = classes
        return folders

    def get_all_instances(self, partition_name='all', with_classes=False):
        """ This returns all instances of a partition of dataset
            Partition can be 'train', 'val', 'test' or 'all'
        """
        instances = list()
        if partition_name == 'all':
            partitions = (self.train_folders, self.val_folders, self.test_folders)
        elif partition_name == 'train':
            partitions = (self.train_folders, )
        elif partition_name == 'test':
            partitions = (self.test_folders,)
        elif partition_name == 'val':
            partitions = (self.val_folders, )
        else:
            raise Exception('The argument partition_name should be in one among all, val, test or train!')

        instance_to_class = dict()
        class_ids = dict()
        class_id = 0
        for partition in partitions:
            for class_name, items in partition.items():
                if class_name not in class_ids:
                    class_ids[class_name] = class_id
                    class_id += 1

                for item in items:
                    instances.append(item)
                    instance_to_class[item] = class_name

        if with_classes:
            return instances, instance_to_class, class_ids

        return instances


class CropDiseaseDatabase(Database):
    def __init__(self, raw_data_address, random_seed=-1):
        super(CropDiseaseDatabase, self).__init__(
            raw_data_address,
            os.getcwd() + '/dataset/data/CropDiseases',
            random_seed=random_seed,
        )

    def get_train_val_test_folders(self):
        dataset_folders = list()
        for dataset_type in ('train', 'test'):
            dataset_base_address = os.path.join(self.database_address, dataset_type)
            folders = [
                os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)
                ]
            dataset_folders.append(folders)

        return dataset_folders[0], dataset_folders[1], dataset_folders[1]

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function
        
    def get_input_shape(self):
        return 84, 84, 3

    def preview_image(self, image_path):
        image = Image.open(image_path)
        return image


class EuroSatDatabase(Database):
    def __init__(self, raw_data_address, random_seed=-1):
        super(EuroSatDatabase, self).__init__(
            raw_data_address,
            os.getcwd() + '/dataset/data/EuroSAT',
            random_seed=random_seed
        )

    def get_train_val_test_folders(self):
        base = os.path.join(self.database_address, '2750')
        folders = [os.path.join(base, folder_name) for folder_name in os.listdir(base)]

        return folders, folders, folders

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function

    def get_input_shape(self):
        return 64, 64, 3

    def preview_image(self, image_path):
        image = Image.open(image_path)
        return image


class ISICDatabase(Database):
    def __init__(self, raw_data_address, random_seed=-1):
        super(ISICDatabase, self).__init__(
            raw_data_address,
            os.getcwd() + '/dataset/data/ISIC',
            random_seed=random_seed
        )

    def get_train_val_test_folders(self):
        gt_file = os.path.join(
            self.database_address,
            'ISIC2018_Task3_Training_GroundTruth',
            'ISIC2018_Task3_Training_GroundTruth.csv'
        )
        content = pd.read_csv(gt_file)
        class_names = list(content.columnms[1:])
        images = list(content.iloc[:, 0])
        labels = np.array(content.iloc[:, 1:])
        labels = np.argmax(labels, axis=1)

        classes = dict()
        for class_name in class_names:
            classes[class_name] = list()
        
        for image, label in zip(images, labels):
            classes[class_names[label]].append(
                os.path.join(self.database_address,'ISIC2018_Task3_Training_Input', image + '.jpg')
            )
        
        return classes, classes, classes

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function

    def get_input_shape(self):
        return 84, 84, 3

    def preview_image(self, image_path):
        image = Image.open(image_path)
        return image


class ChestXRay8Database(Database):
    def __init__(self, raw_data_address, random_seed=-1):
        super(ChestXRay8Database, self).__init__(
            raw_data_address, 
            os.getcwd() + '/dataset/data/chestX',
            random_seed=random_seed
        )

    def get_train_val_test_folders(self):
        image_paths = dict()

        for folder_name in os.listdir(self.database_address):
            if os.path.isdir(os.path.join(self.database_address, folder_name)):
                base_address = os.path.join(self.database_address, folder_name)
                for item in os.listdir(os.path.join(base_address, 'images')):
                    image_paths[item] = os.path.join(base_address, 'images', item)

        gt_file = os.path.join(self.database_address, 'Data_Entry_2017.csv')
        class_names = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"
        ]

        content = pd.read_csv(gt_file)
        images = list(content.iloc[:, 0])

        labels = np.asarray(content.iloc[:, 1])

        classes = dict()
        for class_name in class_names:
            classes[class_name] = list()

        for image, label in zip(images, labels):
            label = label.split("|")
            if(
                len(label) == 1 and 
                label[0] != "No Finding" and
                label[0] != "Pneumonia" and 
                label[0] in class_names
            ):
                classes[label[0]].append(image_paths[image])

        return classes, classes, classes

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_png(tf.io.read_file(example_address), channels=3)
            image = tf.image.resize(image, self.get_input_shape()[:2])
            image = tf.cast(image, tf.float32)
            return image / 255.

        return parse_function

    def get_input_shape(self):
        return 84, 84, 3

    def preview_image(self, image_path):
        image = Image.open(image_path)
        return image