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

        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()
        self.input_shape = self.get_input_shape()

    @abstractmethod
    def get_class(self):
        pass

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

    def get_all_instances(self, partition_name='all', with_classes=False):
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

    def load_dumped_features(self, name: str):
        dir_path = os.path.join(self.database_address, name)
        if not os.path.exists(dir_path):
            raise Exception('Requested features are not dumped.')

        files_names_address = os.path.join(dir_path, 'files_names.npy')
        features_address = os.path.join(dir_path, 'features.npy')
        all_files = np.load(files_names_address)
        features = np.load(features_address)

        return all_files, features

    def dump_vgg19_last_hidden_layer(self, partition: str):
        base_model = tf.keras.applications.VGG19(weights='imagenet')
        model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.layers[24].output)

        self.dump_features(
            partition,
            'vgg19_last_hidden_layer',
            model,
            (224, 224),
            4096,
            tf.keras.applications.vgg19.preprocess_input
        )

    def dump_features(
            self,
            dataset_partition: str,
            name: str,
            model: tf.keras.models.Model,
            input_shape: Tuple,
            feature_size: int,
            preprocess_fn
    ):

        if dataset_partition == 'train':
            files_dir = self.train_folders
        elif dataset_partition == 'val':
            files_dir = self.val_folders
        elif dataset_partition == 'test':
            files_dir = self.test_folders
        else:
            raise Exception('Pratition is not train val or test!')

        assert (dataset_partition in ('train', 'val', 'test'))

        dir_path = os.path.join(self.database_address, f'{name}_{dataset_partition}')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        all_files = list()

        for class_name in files_dir:
            all_files.extend([os.path.join(class_name, file_name) for file_name in os.listdir(class_name)])

        files_names_address = os.path.join(dir_path, 'files_names.npy')
        np.save(files_names_address, all_files)

        features_address = os.path.join(dir_path, 'features.npy')

        n = len(all_files)
        m = feature_size
        features = np.zeros(shape=(n, m))

        begin_time = datetime.now()

        for index, sampled_file in enumerate(all_files):
            if index % 1000 == 0:
                print(f'{index}/{len(all_files)} images dumped')

            img = tf.keras.preprocessing.image.load_img(sampled_file, target_size=(input_shape[:2]))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            if preprocess_fn is not None:
                img = preprocess_fn(img)

            features[index, :] = model.predict(img).reshape(-1)

        np.save(features_address, features)
        end_time = datetime.now()
        print('Features dumped')
        print(f'Time to dump features: {str(end_time - begin_time)}')

    def get_confusion_matrix(self, name: str, partition: str = 'train'):
        if partition == 'train':
            folders = self.train_folders
        elif partition == 'val':
            folders = self.val_folders
        elif partition == 'test':
            folders = self.test_folders
        else:
            raise Exception('Partition should be one of the train, val or test.')

        class_ids = {class_name[class_name.rindex('/') + 1:]: i for i, class_name in enumerate(folders)}
        dir_path = os.path.join(self.database_address, name)
        confusion_matrix_path = os.path.join(dir_path, 'confusion_matrix.npy')
        if os.path.exists(confusion_matrix_path):
            confusion_matrix = np.load(confusion_matrix_path)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            knn_model = KNeighborsClassifier()

            file_names, features = self.load_dumped_features(name)
            num_instances = len(file_names)
            ys = []

            for file_name in file_names:
                class_path = os.path.dirname(file_name)
                class_name = class_path[class_path.rindex('/') + 1:]
                ys.append(class_ids[class_name])

            sampled_instances = np.random.choice(np.arange(len(file_names)), num_instances, replace=False)
            knn_model.fit(features[sampled_instances, :], np.array(ys)[sampled_instances])

            confusion_matrix = np.zeros(shape=(len(folders), len(folders)))
            for i, y in enumerate(ys):
                if i % 1000 == 0:
                    print(f'classifiying {i} out of {len(ys)} is done.')
                predicted_class = knn_model.predict(features[i, ...].reshape(1, -1))
                confusion_matrix[y, predicted_class] += 1

            print(confusion_matrix)
            np.save(confusion_matrix_path, confusion_matrix)

        return confusion_matrix, class_ids


class CropDiseaseDatabase(Database):
    def __init__(self, raw_data_address, random_seed=-1):
        super(CropDiseaseDatabase, self).__init__(
            raw_data_address,
            os.getcwd() + '/dataset/data/CropDiseases',
            random_seed=random_seed,
        )

    def get_class(self):
        train_dict = defaultdict(list)
        test_dict = defaultdict(list)
        for train_class in self.train_folders:
            for train_image_path in glob(train_class + '/*.*'):
                train_dict[train_class.split('\\')[-1]].append(train_image_path)

        for test_class in self.test_folders:
            for test_image_path in glob(test_dict + '/*.*'):
                test_dict[test_class.split('\\')[-1]].append(test_image_path)

        return train_dict, test_dict, test_dict

    def get_train_val_test_folders(self):
        dataset_folders = list()
        for dataset_type in ('train', 'test'):
            dataset_base_address = os.path.join(self.database_address, dataset_type)
            folders = [os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)]
            dataset_folders.append(folders)

        return dataset_folders[0], dataset_folders[1], dataset_folders[1]

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function
        
    def get_input_shape(self) -> Tuple:
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

    def get_class(self):
        print('get the classes of EuroSAT dataset!')

    def get_train_val_test_folders(self):
        base = os.path.join(self.database_address, '2750')
        folders = [os.path.join(base, folder_name) for folder_name in os.listdir(base)]

        return folders, folders, folders

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (64, 64))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function

    def get_input_shape(self) -> Tuple:
        return 64, 64, 3

    def preview_image(self, image_path):
        image = Image.open(image_path)
        return image
