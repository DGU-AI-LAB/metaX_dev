import tensorflow as tf


def triplet_loss(anchor, positive, negative):    
    positive_dist = dist(anchor, positive)
    negative_dist = dist(anchor, negative)
    print("positive_dist: ", positive_dist, "negative_dist: ", negative_dist)
    loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), 20)
    loss_value = tf.maximum(loss_1, 0.0)
    return loss_value
    
def dist(a, b):
    return tf.reduce_sum(tf.square(tf.subtract(a, b)))

        
     