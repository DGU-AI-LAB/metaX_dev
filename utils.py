import tensorflow as tf
import os
import json
import tqdm
import logging

import numpy as np

def combine_first_two_axes(tensor):
    shape = tensor.shape
    return tf.reshape(tensor, (shape[0] * shape[1], *shape[2:]))


def average_gradients(tower_grads, losses):
    average_grads = list()

    for grads, loss in zip(tower_grads, losses):
        grad = tf.math.reduce_mean(grads, axis=0)
        average_grads.append(grad)

    return average_grads

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('OS Error')

def save_nwaykshot(dataset, save_path, class2num, change_mini_imagenet_cls_name=False):
    if change_mini_imagenet_cls_name:
    # class2name mini imagenet class code -> class name
        class2name = create_mini_imagenet_class2name()
        class2name_total = {**class2name['train'], **class2name['val'], **class2name['test']}

    dataset = list(dataset)
    dataset= np.array(dataset)
    # dataset.shape
    # (N of meta-batch, data&label, meta-batch, support&query, N, K)
    # - meta-batch : Number of N-way K-shot Tasks per outer loop update
    # - number of tasks : N of meta-batch * meta-batch

    # Select only data part
    # (N of meta-batch(steps_per_epoch), meta-batch, support&query, N, K)
    dataset = dataset[:,0,:,:,:, :]
    # Flatten the meta-batches
    # (N of meta-batch * meta-batch, support&query, N, K)
    dataset = dataset.reshape(-1, *dataset.shape[2:])

    json_file = {}
    logging.info("Preparing N-way K-shot json files")
    for i, task in tqdm.tqdm(enumerate(dataset)):
        task_name = 'task{}'.format(i+1)
        # Task
        json_file[task_name] = {'supports' : {},
                            "query"    : {} }
        
        support, query = task
        
        for n_class in support:
            class_path = bytes.decode(n_class[0].numpy())
            classnum = class2num[class_path.split(os.sep)[-2]]
            json_file[task_name]['supports'][classnum] = []

            for k_sample in n_class:
                path = bytes.decode(k_sample.numpy())
                name = path.split(os.sep)[-1]
                if change_mini_imagenet_cls_name:
                    name = class2name_total[path.split(os.sep)[-2]]

                json_file[task_name]['supports'][classnum].append(
                    {'name' : name, 'path' : path}
                )

        for n_class in query:
            # print("n_class")
            class_path = bytes.decode(n_class[0].numpy())
            classnum = class2num[class_path.split(os.sep)[-2]]
            json_file[task_name]['query'][classnum] = []

            for k_sample in n_class:
                path = bytes.decode(k_sample.numpy())
                name = path.split(os.sep)[-1]

                if change_mini_imagenet_cls_name:
                    name = class2name_total[path.split(os.sep)[-2]]
                json_file[task_name]['query'][classnum].append(
                    {'name' : name, 'path' : path}
                )

    logging.info("Start Saving N-way K-shot json file")
    print("the N-way K-Shot JSON file has been saved in ", save_path)
    with open(save_path, 'w') as f:
        json.dump(json_file, f, indent='\t')
    logging.info("Saving completed.")

def create_mini_imagenet_class2name():
    
    base = os.path.join(os.getcwd(), 'dataset/data/mini_imagenet'.replace("/", os.sep))
    train_path = os.path.join(base, 'class_names_train.txt')
    test_path = os.path.join(base, 'class_names_test.txt')
    val_path = os.path.join(base, 'class_names_val.txt')

    with open(train_path, 'r', encoding='utf-8') as f:
        train_class_names = f.readlines()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_class_names = f.readlines()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_class_names = f.readlines()


    train_class_names = [i.strip().split(" ") for i in train_class_names]
    test_class_names = [i.strip().split(" ") for i in test_class_names]
    val_class_names = [i.strip().split(" ") for i in val_class_names]

    train_class2name = { k : v for k, v in train_class_names}
    test_class2name = { k : v for k, v in test_class_names}
    val_class2name = { k : v for k, v in val_class_names}

    mini_imagenet_class2name = {'train' : train_class2name, 'val' : val_class2name, 'test' : test_class2name}
    return mini_imagenet_class2name