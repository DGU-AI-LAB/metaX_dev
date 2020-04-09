import numpy as np
import random
import tensorflow as tf
from shseo_load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers
from tensorflow.nn import softmax_cross_entropy_with_logits
import time
import copy
from matplotlib import pyplot as plt
from util import visualization, loss_function

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.lstm_layer1 = tf.keras.layers.LSTM(units=128, return_sequences=True)
        self.lstm_layer2 = tf.keras.layers.LSTM(units=self.num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        
        B, K, N, I= input_images.shape
        
        # First K examples of data+labels 
        data_train = tf.concat([input_images[:,0:-1,:,:], input_labels[:,0:-1,:,:]], axis=3)
        # Last 1 examples of data+zeros 
        data_test = tf.concat([input_images[:,-1:,:,:], tf.zeros_like(input_labels)[:,-1:,:,:]], axis=3)
        
        input_data = tf.concat([data_train, data_test], axis=1) # [B, K+1, N, I+N]

        # reshape input data for matching lstm input shape
        reshaped_input_data = tf.reshape(input_data, [-1, K*N, I+N])
        
        # LSTM layers
        hidden_x = self.lstm_layer1(reshaped_input_data)
        out = self.lstm_layer2(hidden_x)
        
        # reshape output
        reshaped_out = tf.reshape(out, [-1, K, N, N])
        
        return reshaped_out
    
    def grad_function(self, images, labels):
        with tf.GradientTape() as tape:
            preds = self(images, labels)
            ce_loss = loss_function(preds, labels)

        grads = tape.gradient(ce_loss, self.trainable_variables)

            
        return grads, ce_loss
    
    def train(self, FLAGS, data_generator):
        # Set GPU options
                
        optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
        accuracy_metric = tf.keras.metrics.Accuracy()
        
        batch_type = "train"
        batch_size = FLAGS.meta_batch_size
        shuffle = FLAGS.shuffle
        
        # record history
        meta_train_losses = []
        meta_test_losses = []
        meta_test_accuracy = []
        steps = []
        
        for step in range(FLAGS.training_step):
            # load data
            
            meta_train_images, meta_train_labels  = data_generator.sample_batch(batch_type=batch_type, batch_size=batch_size, shuffle=shuffle)
            
            # train phase
            grads, train_loss = self.grad_function(self, meta_train_images, meta_train_labels)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            
            if step % FLAGS.visualization_step == 0:
                print()
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                # meta test data sampling
                meta_test_images, meta_test_labels = data_generator.sample_batch(batch_type='test', batch_size=100)
                
                # inference
                preds = self.call(meta_test_images, meta_test_labels)
                
                # calculate train and test loss
                test_loss = loss_function(preds, meta_test_labels)
                print("Train Loss: {:.4f}".format(train_loss.numpy()), "Test Loss: {:.4f}".format(test_loss.numpy()))
                
                # claculate accuracy
                argmax_preds = tf.math.argmax(preds[:, -1, :, :], 2)
                argmax_meta_test_labels = tf.math.argmax(meta_test_labels[:, -1, :, :], 2)
                _ = accuracy_metric.update_state(y_true=argmax_meta_test_labels, y_pred=argmax_preds)
                acc = accuracy_metric.result().numpy()
                print("Test Accuracy: {:.4f}".format(acc))
                if step != 0:
                    end_time = time.time()
                    print("Elapsed Time: {:.4f}sec".format(end_time-start_time))
                
                # record history
                meta_train_losses.append(train_loss.numpy())
                meta_test_losses.append(test_loss.numpy())
                meta_test_accuracy.append(acc)
                steps.append(step)
                
                # visualization
                visualization(FLAGS, meta_train_losses, meta_test_losses, meta_test_accuracy, steps)
                
                # reset start time
                start_time = time.time()