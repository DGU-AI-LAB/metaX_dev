import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, ReLU, Lambda
from tensorflow.python.keras import Model
from loss_function_tf import triplet_loss

class ConvBlock(Model):
    def __init__(self, filters, kernel_size, input_shape = None):
        super(ConvBlock, self).__init__()
        
        if input_shape is not None:
            self.conv2d = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'valid', activation = 'relu', input_shape = input_shape)
        else:
            self.conv2d = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'valid', activation = 'relu')
        self.bn = BatchNormalization()
            
    def call(self, x):        
        conv_x = self.conv2d(x)
        bn_x = self.bn(conv_x)
        
        return bn_x