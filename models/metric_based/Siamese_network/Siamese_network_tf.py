import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, ReLU, Lambda
from tensorflow.python.keras import Model
from random import randint

from loss_function_tf import triplet_loss, dist
from Siamese_block_tf import ConvBlock
from data_generator_tf import DataGenerator

class SiameseNetwork(Model):
    def __init__(self, way = 2, shot = 1, input_shape = (105, 105, 1)):
        super(SiameseNetwork, self).__init__()
        self.way = way
        self.shot = shot
        
        self.conv1 = ConvBlock(filters = 64, kernel_size = 10, input_shape=input_shape)
        self.conv2 = ConvBlock(filters = 128, kernel_size = 7)
        self.conv3 = ConvBlock(filters = 128, kernel_size = 4)
        self.conv4 = ConvBlock(filters = 256, kernel_size = 4)
        self.maxpool = MaxPool2D((2, 2))
        self.dense = Dense(4096, activation='sigmoid')
                
    def call(self, x):  
        z = self.conv1(x)
        z = self.maxpool(z)
        z = self.conv2(z)
        z = self.maxpool(z)
        z = self.conv3(z)
        z = self.maxpool(z)
        z = self.conv4(z)
        result = self.dense (z)    
        return result
    
    def train_model(self, train_dataset, num_epoch, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        train_dataset.make_data_tensor()
        
        for epoch in range(1, num_epoch+1):
            total_loss = 0
            losses = []

            datas = train_dataset.next()
            support_batch, support_label_batch, query_batch, query_label_batch = zip(*datas)
                
            support_batch = tf.stack(support_batch)
            query_batch = tf.stack(query_batch)

            for support, query  in zip(support_batch,
                                       query_batch):
                label = randint(0, support.shape[0] - 1)
                anchor = query[label]
                
                positive = support[label]
                negative = support[(1+ label)%2]
                
                # loss_value, grads = grad(self, anchor, positive, negative)   
                loss_value, grads = self.grad(anchor, positive, negative)   
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                
            if epoch%10 == 0:
                print("Epoch {:03d}: Loss: {:.3f}".format(epoch, loss_value))
                
    def grad(self, anchor, positive, negative):
        with tf.GradientTape() as tape:
            # y_anchor, y_positive, y_negative = model.call(anchor), model.call(positive), model.call(negative)   
            # x = tf.concat([anchor, positive, negative], axis = 1)
            x = tf.stack([anchor, positive, negative])
            y = self.call(x)
            y_anchor, y_positive, y_negative = y[0], y[1], y[2]
            loss_value = triplet_loss(y_anchor, y_positive, y_negative)
        return loss_value, tape.gradient(loss_value, self.trainable_variables)
                    
    def test_model(self, test_dataset, num_epoch):
        test_dataset.make_data_tensor(train = False)
        
        correct = 0
        total = 0
        
        for epoch in range(1, num_epoch+1):

            datas = test_dataset.next()
            support_batch, support_label_batch, query_batch, query_label_batch = zip(*datas)
                
            support_batch = tf.stack(support_batch)
            query_batch = tf.stack(query_batch)

            for support, query in zip(support_batch,
                                      query_batch):
                label = randint(0, support.shape[0] - 1)                
                query = query[label]
                
                x = tf.concat([[query], support], axis = 0)
                
                y = self.call(x)                
                y_query = y[0]
                y_support = y[0:]
                
                dists = tf.stack([dist(y_query, y_s) for y_s in y_support])
                min_index = tf.argmin(dists)
                
                total += 1
                if label == min_index:
                    correct += 1
                
        return correct/total
    
    def predict(self, support, query):
        support = tf.stack(support)
        query = tf.stack(query)

         x = tf.concat([[query], support], axis = 0)
                
        y = self.call(x)                
        y_query = y[0]
        y_support = y[0:]
                
        dists = tf.stack([dist(y_query, y_s) for y_s in y_support])
        pred = tf.argmin(dists)
        
        return pred