import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import glob
import shutil
import numpy as np
import random
import tqdm
from PIL import Image
tf.enable_eager_execution()

class DataGenerator():
    def __init__(self, image_size, N, K, batch_size):
        self.image_size = image_size
        self.datasetload()
        self.split_data()
        self.N = N
        self.K = K
        self.batch_size = batch_size

    def datasetload(self):
        if os.path.isdir("metaX/data/downloads"):
            print("Dataset already exists")
            download = False
        else:
            print("Dataset Doesn't exist, Download the Dataset")
            download = True
            _ = tfds.load( name="omniglot", as_supervised=True,
        data_dir="metaX\\data\\", download=download, batch_size=-1, shuffle_files=False)
        if os.path.isdir("metaX/data/omniglot_resized"):
            print('already resized')
        else:
            print("Start to resizing images")

            super_list = glob.glob("metaX/data/downloads/extracted/*")
            super_list = [j for i in super_list if "smal" not in i for j in glob.glob(i + "/*/*")]  # super class

            image_path = 'metaX/data/omniglot_resized/'
            os.mkdir(image_path)
            for super_dir in super_list:
                # print(dir) # full path
                super = image_path + super_dir.split('\\')[-1] + '/'
                os.mkdir(super)
                # print(dir.split('\\')[-1]) # super class
                sublist = glob.glob(super_dir + '/*')
                for sub_dir in range(len(sublist)):
                    sub = super + sublist[sub_dir].split('\\')[-1] + '/'
                    os.mkdir(sub)
                    png_list = glob.glob(sublist[sub_dir] + '/*')
                    for png_name in range(len(png_list)):
                        # print(png_list[png_name].split('\\')[-1]) # png name
                        png = sub + png_list[png_name].split('\\')[-1]
                        shutil.move(png_list[png_name], png)

            print("end up moving images")
            image_path = image_path + '*/*/'
            all_images = glob.glob(image_path + '*')

            i = 0
            print("resizing images")
            im = Image.open(all_images[0])
            if np.array(im).shape != self.image_size:
                for image_file in all_images:
                    im = Image.open(image_file)
                    im = im.resize(self.image_size, resample=Image.LANCZOS)
                    im.save(image_file)
                    i += 1

                    if i % 200 == 0:
                        print(i)
            del im

    def split_data(self):
        data_folder = 'metaX/data/omniglot_resized'

        character_folders = [os.path.join(data_folder, family, character) \
                             for family in os.listdir(data_folder) \
                             if os.path.isdir(os.path.join(data_folder, family)) \
                             for character in os.listdir(os.path.join(data_folder, family))]
        # random.seed(2)
        random.shuffle(character_folders)
        self.num_train = 1200
        self.num_test = len(character_folders) - self.num_train

        self.metatrain_character_folders = character_folders[:self.num_train]
        self.metatest_character_folders = character_folders[self.num_train:]
        rotations = [0, 90, 180, 270]

        print('metatrain_character_folders' , len(self.metatrain_character_folders))
        print('metaval_character_folders' , len(self.metatest_character_folders))

    def make_data_tensor(self, train=True):
        def read_image(path):
            image = np.array(Image.open(path))
            return image.reshape(self.image_size[0], self.image_size[1], 1)

        def _read_py_function(path):
            image = read_image(path)
            return image.astype(np.int32)

        def _resize_function(image_decoded):
            image_decoded.set_shape([None, None, None])
            # image_resized = tf.image.resize_images(image_decoded, [28, 28])
            image_decoded = tf.cast(image_decoded, tf.float32) / 255.0
            # image_decoded = 1.0 - image_decoded
            return image_decoded

        if train:
            folders = self.metatrain_character_folders
            total_batch = self.num_train
        else:
            folders = self.metatest_character_folders
            total_batch = self.num_test

        all_filenames, all_labels = [], []
        for _ in tqdm.trange(0, total_batch):
            sampled_character_folders = random.sample(folders, self.N)
            random.shuffle(sampled_character_folders)
            labels_and_images = self.get_images(sampled_character_folders, range(self.N), K=self.K,  shuffle=False)
            # # make sure the above isn't randomized order
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)
            all_labels.extend(labels)

        all_filenames = tf.convert_to_tensor(all_filenames)
        dataset = tf.data.Dataset.from_tensor_slices(all_filenames)
        dataset = dataset.map(
            lambda data_list: tf.py_func(_read_py_function, [data_list], [tf.int32]))
        dataset = dataset.map(_resize_function)

        task_image_size = (self.N * self.K) * 2
        batch_image_size = self.batch_size * task_image_size
        # print('Batching images')
        dataset = dataset.batch(batch_image_size)

        support_batch_data, support_batch_labels, query_batch_data, query_batch_labels = [], [], [], []
        # print('Manipulating image data to be right shape')

        # index choices support & query
        query_task_idx = []
        idx_list = np.array(range(task_image_size))
        reshape_idx_list = idx_list.reshape(task_image_size // (self.K*2), self.K*2)

        for class_idx in range(reshape_idx_list.shape[0]):
            query_task_idx.extend(reshape_idx_list[class_idx][:self.K])
        temp_list = set(query_task_idx)
        support_task_idx = [x for x in idx_list if x not in temp_list]

        # task 별로 뽑아내기
        iter = tfe.Iterator(dataset)
        for num_task in range(self.batch_size):
            images = iter.next()
            support_task_images, support_task_labels = [], []
            query_task_images, query_task_labels = [], []
            for support_idx in support_task_idx:
                support_task_images.append(images[support_idx])
                support_task_labels.append(labels[support_idx])
            for query_idx in query_task_idx:
                query_task_images.append(images[query_idx])
                query_task_labels.append(labels[query_idx])

            support_batch_data.append(support_task_images)
            # print(tf.one_hot(support_task_labels, self.N))
            support_batch_labels.append(tf.one_hot(support_task_labels, self.N))
            query_batch_data.append(query_task_images)
            query_batch_labels.append(tf.one_hot(query_task_labels, self.N))

        support_batch_data = tf.stack(support_batch_data)
        support_batch_labels = tf.stack(support_batch_labels)
        query_batch_data = tf.stack(query_batch_data)
        query_batch_labels = tf.stack(query_batch_labels)
        return support_batch_data, support_batch_labels, query_batch_data, query_batch_labels



    def get_images(self, paths, labels, K=None,  shuffle=True):
        if K is not None:
            sampler = lambda x: random.sample(x, K*2)
        else:
            sampler = lambda x: x
        images = [(i, os.path.join(path, image)) \
                  for i, path in zip(labels, paths) \
                  for image in sampler(os.listdir(path))]
        if shuffle:
            random.shuffle(images)
        return images

if __name__ == '__main__':
    n = 5
    k = 1
    batch_size = 32
    image_resize = (28, 28)
    data_generator = DataGenerator(image_resize, N=n, K=k, batch_size=batch_size)
    support_batch_data, support_batch_labels, query_batch_data, query_batch_labels = data_generator.make_data_tensor()
    print(support_batch_data.shape)
    print(support_batch_labels.shape)
    print(query_batch_data.shape)
    print(query_batch_labels.shape)
    import matplotlib.pyplot as plt

    if not os.path.isdir('metaX/support/'):
        os.mkdir('metaX/support/')
    if not os.path.isdir('metaX/query/'):
        os.mkdir('metaX/query/')

    for k in range(batch_size):
        for i in range(support_batch_data.shape[1]):
            plt.imshow(support_batch_data[k][i].numpy().reshape(28, 28), cmap="gray")
            plt.savefig("metaX/support/{}_{}.jpg".format(k, i))
        for j in range(query_batch_data.shape[1]):
            plt.imshow(query_batch_data[k][j].numpy().reshape(28, 28), cmap="gray")
            plt.savefig("metaX/query/{}_{}.jpg".format(k, j))