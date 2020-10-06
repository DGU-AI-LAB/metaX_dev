import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os

from PIL import Image
from collections import defaultdict
from abc import ABC, abstractmethod
from glob import glob


class Database(ABC):
    def __init__(self, 
                 raw_database_address, 
                 database_address, 
                 random_seed=-1, 
    ):
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

    @abstractmethod
    def get_train_val_test_folders(self):
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

    def _get_parse_function(self):
        def parse_function(example_address):
            return example_address
        
        return parse_function
    
    def make_labels_dataset(self, n, k, meta_batch_size, steps_per_epoch, one_hot_labels):
        labels_dataset = tf.data.Dataset.range(n)
        
        if one_hot_labels:
            labels_dataset = labels_dataset.map(lambda example: tf.one_hot(example, depth=n))

        labels_dataset = labels_dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(2 * k),
            cycle_length=n,
            block_length=k
        )
        labels_dataset = labels_dataset.repeat(meta_batch_size)
        labels_dataset = labels_dataset.repeat(steps_per_epoch)
        
        return labels_dataset
   
    def _get_instances(self, k):
        def get_instances(class_dir_address):
            return tf.data.Dataset.list_files(class_dir_address, shuffle=True).take(2 * k)
        
        return get_instances
    
    def keep_keys_with_greater_than_equal_k_items(self, folders_dict, k):
        to_be_removed = list()
        for folder in folders_dict.keys():
            if len(folders_dict[folder]) < k:
                to_be_removed.append(folder)

        for folder in to_be_removed:
            del folders_dict[folder]
    
    def get_dataset(
        self, 
        folders,
        n, 
        k, 
        meta_batch_size,
        one_hot_labels=True, 
        reshuffle_each_iteration=True,
        random_seed=-1,
        dtype=tf.float32,
    ):
        
        def convert_folders_to_list(folders):
            if type(folders) == list:
                classes = dict()
                for folder in folders:
                    instances = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
                    classes[folder] = instances
                folders = classes
                
            return folders

        folders = convert_folders_to_list(folders)
        self.keep_keys_with_greater_than_equal_k_items(folders, k)

        dataset = tf.data.Dataset.from_tensor_slices(sorted(list(folders.keys())))
        steps_per_epoch = len(folders.keys()) // (n * meta_batch_size)
        
        if random_seed != -1:
            dataset = dataset.shuffle(
                buffer_size=len(folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration,
                random_seed=random_seed
            )
            dataset = dataset.interleave(
                self._get_instances(k),
                cycle_length=n,
                block_length=k,
                num_parallel_calls=1
            )
        else:
            dataset = dataset.shuffle(
                buffer_size=len(folders.keys()),
                reshuffle_each_iteration=reshuffle_each_iteration
            )
            dataset = dataset.interleave(
                self._get_instances(k),
                cycle_length=n,
                block_length=k,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
    
        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)
        
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        
        return dataset


class CropDiseaseDatabase(Database):
    def __init__(self, raw_database_address, random_seed=-1):
        super(CropDiseaseDatabase, self).__init__(
            raw_database_address,
            os.getcwd() + '/dataset/data/CropDiseases',
            random_seed=random_seed
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
            image = tf.image.decode_jpeg(tf.io.read_file(example_address), channels=3)
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
            image = tf.image.decode_jpeg(tf.io.read_file(example_address), channels=3)
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function

    def get_input_shape(self):
        return 84, 84, 3

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
        class_names = list(content.columns[1:])

        images = list(content.iloc[:, 0])

        labels = np.array(content.iloc[:, 1:])
        labels = np.argmax(labels, axis=1)

        classes = dict()
        for class_name in class_names:
            classes[class_name] = list()

        for image, label in zip(images, labels):
            classes[class_names[label]].append(
                os.path.join(self.database_address, 'ISIC2018_Task3_Training_Input', image + '.jpg')
            )

        return classes, classes, classes


    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address), channels=3)
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