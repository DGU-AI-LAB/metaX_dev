import tensorflow as tf
import os

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
