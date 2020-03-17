import tensorflow as tf
import random
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from utils import ConvBlock, np_to_tensor, compute_loss


class MAML(Model):
    def __init__(self, units):
        super(MAML, self).__init__()
        self.hidden1 = ConvBlock(filters=64, kernel_size=(3, 3), stride= 2, axis=-1)
        self.hidden2 = ConvBlock(filters=64, kernel_size=(3, 3), stride= 2, axis=-1)
        self.hidden3 = ConvBlock(filters=64, kernel_size=(3, 3), stride= 2, axis=-1)
        self.hidden4 = ConvBlock(filters=64, kernel_size=(3, 3), stride= 2, axis=-1)
        self.final = Dense(units)  # for output layer

    def call(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = tf.reduce_mean(x, [1, 2])
        return self.final(x)

