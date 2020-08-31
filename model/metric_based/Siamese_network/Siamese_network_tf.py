import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, ReLU, Lambda, TimeDistributed
from tensorflow.python.keras import Model
from random import randint
import time

from loss_function_tf import triplet_loss, dist
from Siamese_block_tf import ConvBlock
from data_generator_tf import DataGenerator

class SiameseNetwork(Model):
    def __init__(self, input_shape = (105, 105, 1)):
        super(SiameseNetwork, self).__init__()
        
        self.conv1 = ConvBlock(filters = 64, kernel_size = 10, input_shape=input_shape)
        self.conv2 = ConvBlock(filters = 128, kernel_size = 7)
        self.conv3 = ConvBlock(filters = 128, kernel_size = 4)
        self.conv4 = ConvBlock(filters = 256, kernel_size = 4)
        self.maxpool = MaxPool2D((2, 2))
        self.flatten = Flatten()
        self.dense = Dense(4096, activation='sigmoid')
                
    def call(self, x):  
        z = self.conv1(x)
        z = self.maxpool(z)
        z = self.conv2(z)
        z = self.maxpool(z)
        z = self.conv3(z)
        z = self.maxpool(z)
        z = self.conv4(z)
        z = self.flatten(z)
        result = self.dense(z)    
        return result
    
    def grad(self, anchor, positive, negative):
        with tf.GradientTape() as tape:
            # y_anchor, y_positive, y_negative = model.call(anchor), model.call(positive), model.call(negative)   
            # x = tf.concat([anchor, positive, negative], axis = 1)
            x = tf.concat([anchor, positive, negative], 0)
            y = self.call(x)
            k = (int)(y.shape[0] / 3)
            y_anchor, y_positive, y_negative = y[:k], y[k:2*k], y[2*k:]
            loss_value = triplet_loss(y_anchor, y_positive, y_negative)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)
    
    def train_model(self, train_dataset, num_epoch, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        losses = []
        for epoch in range(1, num_epoch+1):
            train_dataset.make_data_tensor()
            support_batch, support_batch_labels, query_batch, query_batch_labels = train_dataset.next()
                
            for support, support_label, query, query_label  in zip(support_batch,
                                                                   support_batch_labels,
                                                                   query_batch,
                                                                   query_batch_labels):
                
                query_label = tf.stack([np.where(l == 1)[2] for l in query_label])
                support_label = tf.stack([np.where(l == 1)[2] for l in support_label])
                
                anchor_label = np.array([ql[0] for ql in query_label])
                anchor = tf.stack([q[0] for q in query])      
                
                positive_label = [np.argwhere(sl == al) for sl, al in zip(support_label, anchor_label)]
                negative_label = [np.argwhere(sl != al) for sl, al in zip(support_label, anchor_label)]
                
                
                negative = tf.stack([s[nl][0][0] for s, nl in zip(support, negative_label)])
                positive = tf.stack([s[pl][0][0] for s, pl in zip(support, positive_label)])
                
                
                loss_value, grads = self.grad(anchor, positive, negative)   
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                losses.append(loss_value)
            if epoch%10 == 0:
                print("Epoch {:03d}: Loss: {:.3f} Time: {}\n".format(epoch, sum(losses)/len(losses), time.strftime('%X', time.localtime(time.time()))))
                losses = []
                
      
    def predict(self, support, query):
        x = tf.concat([[query], support], axis = 0)
                
        y = self.call(x)                
        y_query = y[0]
        y_support = y[0:]
                
        dists = tf.stack([dist(y_query, y_s) for y_s in y_support])
        pred = tf.argmin(dists)
        
        return pred