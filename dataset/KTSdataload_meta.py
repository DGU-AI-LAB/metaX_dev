import numpy as np
import os
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class KTS_datset:
    def __init__(self, dir = "./KTS/total", n_way = 5, k_shot = 1, batch_size = 32, image_size = (32, 32)):
        self.dir = dir        
        self.n_way = n_way
        self.k_shot = k_shot
        self.batch_size = batch_size
        self.image_size = image_size
        self.index = 0
        
        self.split_labels()
        
    def split_labels(self):
        self.labels = os.listdir(self.dir)
        random.shuffle(self.labels)         
        self.train_labels = self.labels[:5]
        self.valid_labels = self.labels[5:7]
        self.test_labels = self.labels[7:]
        print("dataset splited")
    
    def set_mode(self, mode = "train"):
        if mode == "train":
            self.using_labels = self.train_labels
        elif mode == "valid":
            self.using_labels = self.valid_labels
        elif mode == "test":
            self.using_labels = self.test_labels
        else:
            return None  
              
        self.read_images()
        self.len = sum([len(self.images[label]) for label in self.using_labels])
        print("read", self.len, "images from", self.dir)
        print("labels for", mode, "are ", self.using_labels)    
        
    def read_images(self, ):
        self.images = {}
        for label in self.using_labels:
            file_names = os.listdir(self.dir + "/" + label + "/images")
            self.images[label] = [Image.open(self.dir + "/" + label + "/images/" + image_name).resize(self.image_size, resample=Image.LANCZOS) for image_name in file_names]
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == self.len:
            raise StopIteration
        elif self.index + self.batch_size < self.len:
            return_size = self.batch_size
            self.index += self.batch_size
        else:
            return_size = self.len - self.index
            self.index = self.len
                        
        support_set = []
        query_set = []
        support_labels = []
        query_labels = []
        
        for i in range(return_size):
            labels = random.sample(self.using_labels, self.n_way)
            support_labels.append(labels)
            query_labels.append(labels)
            
            task_support = []
            task_query = []
            for label in labels:
                image_names = random.sample(self.images[label], self.k_shot * 2)
                support_names = image_names[:self.k_shot]
                query_names = image_names[self.k_shot:]
                
                task_support.append(support_names)
                task_query.append(query_names)
            
            support_set.append(task_support)
            query_set.append(task_query)
                    
        return support_set, support_labels, query_set, query_labels
    
ds = KTS_datset(batch_size=100)
ds.set_mode("train")
for ss, sl, qs, ql in ds:
    # print("ㅁ", end = '')
    print(sl[0][0])
    ss[0][0][0].show()
    break
