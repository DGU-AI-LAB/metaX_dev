import tensorflow as tf  
import tqdm 
import os
import glob
import shutil
import random
from tensorflow.python.platform import flags

import os
import numpy as np
from tqdm import trange

FLAGS = flags.FLAGS

class MiniImageNetDataLoader:
    def __init__(self, image_dir = './data/miniImagenet', batch_size = 16, k_shot = 1, n_way = 5, image_size = (84, 84), shuffle_images = False):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.n_way = n_way
        self.image_size = image_size
        self.shuffle_images = shuffle_images
        
        #self.split_labels()
        self.process_images()

    def split_labels(self):
        data_dir = './data/miniImagenet'
        train_folder = './images/train'
        val_folder = './iamges/val'
        test_folder = './images/test'

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        self.data_base_dir = data_dir + str(self.batch_size) + 'batch_' + str(self.k_shot) + 'shot_' + str(self.n_way) + 'way_' + str(self.image_size) + 'image_' + str(self.shuffle_images) + '/'
        if not os.path.exists(self.data_base_dir):
            os.mkdir(self.data_base_dir)
        
        if not os.path.exists(os.path.join(data_dir, train_folder)):
            os.makedirs(os.path.join(data_dir, train_folder))
            
        if not os.path.exists(os.path.join(data_dir, val_folder)):
            os.makedirs(os.path.join(data_dir, val_folder))
            
        if not os.path.exists(os.path.join(data_dir, test_folder)):
            os.makedirs(os.path.join(data_dir, test_folder))         
             
        self.metatrain_folders = [os.path.join(data_dir, train_folder, label) \
            for label in os.listdir(os.path.join(data_dir, train_folder)) \
            if os.path.isdir(os.path.join(data_dir, train_folder, label)) \
            ]
        self.metaval_folders = [os.path.join(data_dir, val_folder, label) \
            for label in os.listdir(os.path.join(data_dir, val_folder)) \
            if os.path.isdir(os.path.join(data_dir, val_folder, label)) \
            ]
        self.metatest_folders = [os.path.join(data_dir, test_folder, label) \
            for label in os.listdir(os.path.join(data_dir, test_folder)) \
            if os.path.isdir(os.path.join(data_dir, test_folder, label)) \
            ]
        print("dataset splited")

    def process_images(self):
        path_to_images = 'images/'

        all_images = glob.glob(path_to_images + '*')

        for i, image_file in enumerate(all_images):
            im = Image.open(image_file)
            im = im.resize((84, 84), resample=Image.LANCZOS)
            im.save(image_file)
            if i % 500 == 0:
                print(i)

        for datatype in ['train', 'val', 'test']:
            os.system('mkdir ' + datatype)

            with open(datatype + '.csv', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                last_label = ''
                for i, row in enumerate(reader):
                    if i == 0:  # skip the headers
                        continue
                    label = row[1]
                    image_name = row[0]
                    if label != last_label:
                        cur_dir = datatype + '/' + label + '/'
                        os.system('mkdir ' + cur_dir)
                        last_label = label
                    os.system('mv images/' + image_name + ' ' + cur_dir)

    def split_data():
        metatrain_folder = ('./data/miniImagenet/train') 

        if FLAGS.test_set: 
            metaval_folder = ('./data/miniImagenet/test') 
        else: 
            metaval_folder = ('./data/miniImagenet/val') 
        
        metatrain_folders = [os.path.join(metatrain_folder, label) \ 
                             for label in os.listdir(metatrain_folder) \ 
                             if os.path.isdir(os.path.join(metatrain_folder, label)) \ 
                             ] 
        metaval_folders = [os.path.join(metaval_folder, label) \ 
            for label in os.listdir(metaval_folder) \ 
            if os.path.isdir(os.path.join(metaval_folder, label)) \
            ] 
        self.metatrain_character_folders = metatrain_folders 
        self.metaval_character_folders = metaval_folders 
        rotations = [0]

    def get_images(self, paths, labels, num_images, shuffle = True):
        if num_images is not None:
            sampler = lambda x: random.sample(x, num_images)
        else:
            sampler = lambda x: x
        images = [(i, os.path.join(path, image)) \
            for i, path in zip(labels, paths) \
                for image in sampler(os.listdir(path))]
        if shuffle:
            random.shuffle(images)
        return images
            
    def make_data_tensor(self, train=True): 
        def read_image(path):
            image = np.array(Image.open(path.numpy()))
            return image.reshape(self.image_size[0], self.image_size[1], 1)

        if train: 
            folders = self.metatrain_character_folders 
            num_total_batches = 200000 
        else: 
            folders = self.metaval_character_folders 
            num_total_batches = 600 

        print('Generating filenames')
        all_filenames = []
        for _ in range(num_total_batches):
            sampled_character_folders = random.sample(folders, self.n_way)
            random.shuffle(sampled_character_folders)
            labels_and_images = get_images(sampled_character_folders, range(self.n_way), nb_samples=self.k_shot, shuffle=False)
            labels = [li[0] for li in labels_and_images]
            filenames = [li[1] for li in labels_and_images]
            all_filenames.extend(filenames)

        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print('Generating image processing ops')
        image_reader = tf.WholeFileReader()
        image = tf.image.decode_jpeg(image_file, channels=3)
        image.set_shape((self.img_size[0],self.img_size[1],3))
        image = tf.reshape(image, [self.dim_input])
        image = tf.cast(image, tf.float32) / 255.0

        new_list = tf.concat(new_list, 0)  
            new_label_list = tf.concat(new_label_list, 0)
            all_image_batches.append(new_list)
            all_label_batches.append(new_label_list)
        all_image_batches = tf.stack(all_image_batches)
        all_label_batches = tf.stack(all_label_batches)
        all_label_batches = tf.one_hot(all_label_batches, self.num_classes)
        return all_image_batches, all_label_batches

    def iter(self):
        return self
    
    def next(self):
        if self.index == self.len:
            raise StopIteration
        elif self.index + self.batch_size < self.len:
            return_size = self.batch_size
            self.index += self.batch_size
        else:
            return_size = self.len - self.index
            self.index = self.len
                        
        support_set, query_set, support_labels, query_labels  = [], [], [], []
        
        for i in range(return_size):
            labels = random.sample(self.using_labels, self.n_way)
            support_labels.append(labels)
            query_labels.append(labels)
            
            task_support, task_query = [], []
            for label in labels:
                image_names = random.sample(self.images[label], self.k_shot * 2)
                support_names = image_names[:self.k_shot]
                query_names = image_names[self.k_shot:]
                
                task_support.append(support_names)
                task_query.append(query_names)
            
            support_set.append(task_support)
            query_set.append(task_query)
                    
        return zip(support_set, support_labels, query_set, query_labels)

