import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import numpy as np
import random
tf.enable_eager_execution()

class DataGenerator():
    def __init__(self, image_size, N, meta_lr=1e-4, train_lr=1e-3):
        self.image_size = image_size
        self.N = N
        self.meta_lr = meta_lr
        self.train_lr = train_lr
        
    def initweight():
        
