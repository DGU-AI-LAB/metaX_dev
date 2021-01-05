from abc import ABC, abstractmethod
from PIL import Image
from pickle import dump, load
from tqdm import tqdm
import os, shutil # @ package added
from glob import glob
import gdown, zipfile # 
import random
import utils
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.text import Tokenizer

class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.raw_database_address = raw_database_address
        self.database_address = database_address

        self.prepare_database()
        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()

        self.input_shape = self.get_input_shape()

    @abstractmethod
    def get_class(self):
        pass

    @abstractmethod
    def preview_image(self):
        pass

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def prepare_database(self):
        pass

    @abstractmethod
    def get_train_val_test_folders(self):
        pass

    def check_number_of_samples_at_each_class_meet_minimum(self, folders, minimum):
        for folder in folders:
            if len(os.listdir(folder)) < 2 * minimum:
                raise Exception(
                    f'There should be at least {2 * minimum} examples in each class. Class {folder} does not have that many examples')

    def _get_instances(self, k):
        def get_instances(class_dir_address):
            return tf.data.Dataset.list_files(class_dir_address, shuffle=True).take(2 * k)

        return get_instances

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

    def get_supervised_meta_learning_dataset(
            self,
            folders,
            n,
            k,
            meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
    ):
        for class_name in folders:
            assert (len(os.listdir(class_name)) > 2 * k), f'The number of instances in each class should be larger ' \
                f'than {2 * k}, however, the number of instances in' \
                f' {class_name} are: {len(os.listdir(class_name))}'

        classes = [class_name + '/*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size

        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        # print(len(folders))
        dataset = dataset.shuffle(buffer_size=len(folders), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.interleave(
            self._get_instances(k),
            cycle_length=n,
            block_length=k,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

class MSCOCOKRDatabase(Database):

    def __init__(self, train_address, val_address, test_address):
        self.train_address = train_address
        self.val_address = val_address
        self.test_address = test_address
        os.makedirs(train_address, exist_ok=True)
        os.makedirs(val_address, exist_ok=True)
        os.makedirs(test_address, exist_ok=True)

        # Before update
        # self.train_address = os.path.join(os.getcwd(), train_address)
        # self.test_address = os.path.join(os.getcwd(), test_address)
      
    def preview_image(self, image_path):
        image = Image.open(image_path)
        image.show()
        return image
    
    def get_input_shape(self):
        return 299, 299, 3

    def get_class(self):
        print("ImageCaptioningModel is not a classification model.")

    def get_train_val_test_folders(self):
        train_folders = self.train_address
        val_folders = self.val_address
        test_folders = self.test_address
        
        return train_folders, val_folders, test_folders

    def extract_features(self, directory):
        model = Xception(include_top=False, pooling='avg' )
        features = {}
        for img in os.listdir(directory):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image_shape_temp = image.shape
            if len(image_shape_temp) < 4: # 흑백사진 처리
                image = np.expand_dims(image, axis=3)
                image = np.append(image, np.append(image, image, axis = 3), axis = 3)
            image = image/127.5
            image = image - 1.0
            feature = model.predict(image)
            features[img] = feature
        feature_path = self.train_address + "/features.p"
        dump(features, open(feature_path,"wb"))
        return features


    def dict_to_list(self, descriptions):
        all_desc = []
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc
     
    def create_tokenizer(self, descriptions):
        desc_list = self.dict_to_list(descriptions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(desc_list)
        return tokenizer

    def max_length(self, descriptions):
        desc_list = self.dict_to_list(descriptions)
        return max(len(d.split()) for d in desc_list)

    def load_clean_descriptions(self, filename, photos): 
        file = self.load_doc(filename)
        descriptions = {}
        for line in file.split("\n"):
            words = line.split()
            if len(words)<1 :
                continue
            image, image_caption = words[0], words[1:]
            if image in photos:
                if image not in descriptions:
                    descriptions[image] = []
                desc = '<start> ' + " ".join(image_caption) + ' <end>'
                descriptions[image].append(desc)
        return descriptions
    
    def load_doc(self, filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text

    def load_photos(self, filename):
        file = self.load_doc(filename)
        photos = file.split("\n")[:-1]
        return photos
    
    def prepare_database(self):
        output1 = os.path.join(self.train_address, 'features.p')
        output2 = os.path.join(self.test_address, 'test_images.zip')
        output3 = os.path.join(self.train_address, 'train_images.zip')
        output4 = os.path.join(self.val_address, 'val_images.zip')

        if os.path.exists(output1) and os.path.exists(output2) \
            and os.path.exists(output3) and os.path.exists(output4): 
            return
        print("Download the datasets")
        url1 = 'https://drive.google.com/uc?id=1tGiQmHfuyjbsWJojLzAe72a5nJ-rWkAp'
        url2 = 'https://drive.google.com/uc?id=1iE2wZ94f6LF8a6zdVshLisTTS8d4Urnb'
        url3 = 'https://drive.google.com/uc?id=1M9sUpHs9iF_G_-JlUBYhCSJjxLOJ75M4'
        url4 = 'https://drive.google.com/uc?id=1cAAzLdmTz_lgX9yG0wh8YeDwWzItskpO'

        gdown.download(url1, output1, quiet=False)        
        gdown.download(url2, output2, quiet=False)
        gdown.download(url3, output3, quiet=False)
        gdown.download(url4, output4, quiet=False)
        

        train_zip = zipfile.ZipFile(os.path.join(self.train_address, 'train_images.zip'))
        train_zip.extractall(os.path.join(self.train_address, 'train_images'))
        train_zip.close()

        test_zip = zipfile.ZipFile(os.path.join(self.test_address, 'test_images.zip'))
        test_zip.extractall(os.path.join(self.test_address, 'test_images'))
        test_zip.close()

        val_zip = zipfile.ZipFile(os.path.join(self.val_address, 'val_images.zip'))
        val_zip.extractall(os.path.join(self.val_address, 'val_images'))
        val_zip.close()
