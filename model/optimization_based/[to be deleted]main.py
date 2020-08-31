import tensorflow as tf
import random, time
from utils import np_to_tensor, compute_loss
from maml import MAML
from data_generator import DataGenerator


def copy_model(model, x, K):
    copied_model = MAML(units=K)  # units should be changed
    copied_model.predict(tf.convert_to_tensor(x))
    copied_model.set_weights(model.get_weights())

    return copied_model


def train_maml(model, epochs, dataset, lr_inner=0.01, batch_size=32, log_steps=1000):
    optimizer = tf.keras.optimizers.SGD()
    dataset.make_data_tensor()
    # Initialize pi
    # Line 2
    #  for iteration = 1, 2, ... do
    for epoch in range(epochs):
        total_loss = 0
        losses = []
        start = time.time()

        datas = dataset.next()
        support_batch_data, support_batch_labels, query_batch_data, query_batch_labels = zip(*datas)

        # Line 3
        # Sample tasks t1, t2, ... ,tn
        support_batch_data = tf.stack(support_batch_data)
        support_batch_labels = tf.stack(support_batch_labels)
        query_batch_data = tf.stack(query_batch_data)
        query_batch_labels = tf.stack(query_batch_labels)

        model.call(support_batch_data[0])
        before_weights = model.get_weights()

        for i in range(batch_size): # 32 tasks
            # Step 4
            support_task_data = support_batch_data[i] # (5, 28, 28, 1)
            support_task_labels = support_batch_labels[i]
            query_task_data = query_batch_data[i]
            query_task_labels = query_batch_labels[i]

            # Line 4
            # for i = 1, 2, ..., n do
            for k in range(10): # k step (inner steps)
                model.call(support_task_data)

                with tf.GradientTape() as train_tape:

                    train_loss, _ = compute_loss(model, support_task_data, support_task_labels)
                # Step 6
                gradients = train_tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            after_weights = model.get_weights()

            if i == 0:
                after_weights_sum = np.array(after_weights)
            else:
                after_weights_sum += np.array(after_weights)


        # after_weights_sum = np.array(after_weights_sum) / 10
        # after_weights_sum = np_to_tensor(after_weights_sum)

        print(tf.norm(after_weights_sum[0] - before_weights[0]))
        model.set_weights([before_weights[j] + (after_weights_sum[j] - before_weights[j]) * 1e-4 for j in range(len(model.weights))])

        total_loss += train_loss
        if i == batch_size-1:
            loss = total_loss / (i + 1.0)

        losses.append(loss)

        print('epoch {}: loss = {}, Time to run {}'.format(epoch, np.mean(losses), time.time() - start))




def main():
    # dataset setting
    model = MAML(units=10)  # units should be changed
    # train_maml(model, ...)  # in progress


if __name__ == "__main__":

    # the under code for Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED issue
    from tensorflow.compat.v1 import ConfigProto
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # tf.enable_eager_execution()
    import numpy as np
    import tensorflow as tf

    n = 5
    k = 10
    batch_size = 10
    image_resize = (28, 28)
    data_generator = DataGenerator(image_resize, N=n, K=k, batch_size=batch_size)
    # support_batch_data, support_batch_labels, query_batch_data, query_batch_labels = data_generator.make_data_tensor()
    data_generator.make_data_tensor()
    datas = data_generator.next()


    model = MAML(units=n)
    train_maml(model, epochs=1000, dataset=data_generator, lr_inner=0.001, batch_size=batch_size, log_steps=1000)



    # for i in range(batch_size):
    #     print(model.call(support_batch_data[i]).shape)

    # FLAGS = flags.FLAGS
    # main()