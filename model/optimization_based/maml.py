import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
import pickle
from tqdm import tqdm
from utils import combine_first_two_axes, createFolder
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization
from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
from model.LearningType import MetaLearning



class OmniglotModel(tf.keras.Model):
    name = 'OmniglotModel'

    def __init__(self, num_classes):
        super(OmniglotModel, self).__init__(name='omniglot_model')

        self.conv1 = tf.keras.layers.Conv2D(64, 3, name='conv1', strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(64, 3, name='conv2', strides=(2, 2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, name='conv3', strides=(2, 2), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(64, 3, name='conv4',  strides=(2, 2), padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(num_classes, activation=None, name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reduce_mean(c4, [1, 2])
        f = self.flatten(c4)
        out = self.dense(f)

        return out


class MiniImagenetModel(tf.keras.Model):
    name = 'MiniImagenetModel'

    def __init__(self, num_classes):
        super(MiniImagenetModel, self).__init__(name='mini_imagenet_model')
        self.conv1 = tf.keras.layers.Conv2D(32, 3, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = tf.keras.layers.Conv2D(32, 3, name='conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = tf.keras.layers.Conv2D(32, 3, name='conv4')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        self.dense = Dense(num_classes, activation=None, name='dense')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        batch_normalized_out = self.max_pool(batch_normalized_out)
        return tf.keras.activations.relu(batch_normalized_out)

    def call(self, inputs, training=False):
        image = inputs
        c1 = self.conv_block(image, self.conv1, self.bn1, training=training)
        c2 = self.conv_block(c1, self.conv2, self.bn2, training=training)
        c3 = self.conv_block(c2, self.conv3, self.bn3, training=training)
        c4 = self.conv_block(c3, self.conv4, self.bn4, training=training)
        c4 = tf.reshape(c4, [-1, np.prod([int(dim) for dim in c4.get_shape()[1:]])])
        out = self.dense(c4)

        return out


class ModelAgnosticMetaLearning(MetaLearning):
    def __init__(
            self,
            args,
            database,
            network_cls,
            base_dataset_path, # 20.10.13. for ui output file path 
            least_number_of_tasks_val_test=-1,
            # Make sure the validaiton and test dataset pick at least this many tasks.
            clip_gradients=False
        ):
        # Common Meta-Learning hyperparameters
        super(ModelAgnosticMetaLearning, self).__init__(
                        args,
                        database,
                        network_cls,
                        least_number_of_tasks_val_test=-1,
                        # Make sure the validaiton and test dataset pick at least this many tasks.
                        clip_gradients=False
        )


        # MAML hyperparameters
        self.num_steps_ml = args.num_steps_ml
        self.lr_inner_ml = args.lr_inner_ml
        self.num_steps_validation = args.num_steps_validation

        # Initialize the MAML
        self.model(tf.zeros(shape=(self.n * self.k, *self.database.input_shape)))
        self.updated_models = list()
        for _ in range(self.num_steps_ml + 1):
            updated_model = self.network_cls(num_classes=self.n)
            updated_model(tf.zeros(shape=(self.n * self.k, *self.database.input_shape)))
            self.updated_models.append(updated_model)

        # Loging and Model Saving setting
        '''
        self.save_after_epochs                : Interval epoch for saving the model
        self.log_train_images_after_iteration : Interval iteration for writting the classified image on Tensorboard
        self.report_validation_frequency      : Interval epoch for reporting validation frequency

        self._root                            : Root path
        self.train_log_dir                    : Train log path 
        '''
        self.save_after_epochs = args.save_after_epochs                               # type : int
        self.log_train_images_after_iteration = args.log_train_images_after_iteration # type : int
        self.report_validation_frequency = args.report_validation_frequency           # type : int

        # self._root = self.get_root()                                                         # type : string
        # self._root = os.path.join(os.getcwd(), 'dataset/data/ui_output', 'maml_{}'.format(self.args.benchmark_dataset), 'step3')
        self._root = base_dataset_path
        print(self._root)
        self.train_log_dir = os.path.join(self._root, self.get_config_info(), 'train') # type : string
        os.makedirs(self.train_log_dir, exist_ok=True)

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'val')
        os.makedirs(self.val_log_dir, exist_ok=True)

        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models') # 20.09.03
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.test_log_dir = os.path.join(self._root, self.get_config_info(), 'test')
        os.makedirs(self.test_log_dir, exist_ok=True)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
        self.test_count = 0 # 20.10.13. added for test


    def get_root(self):
        return os.path.dirname(__file__)

    def get_train_dataset(self):
        dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.train_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=self.meta_batch_size
        )
        return dataset

    def get_val_dataset(self):
        val_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.val_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
            reshuffle_each_iteration=True
        )
        steps_per_epoch = max(val_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        val_dataset = val_dataset.repeat(-1)
        val_dataset = val_dataset.take(steps_per_epoch)
        setattr(val_dataset, 'steps_per_epoch', steps_per_epoch)
        return val_dataset

    def get_test_dataset(self):
        test_dataset = self.database.get_supervised_meta_learning_dataset(
            self.database.test_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
        )
        steps_per_epoch = max(test_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        test_dataset = test_dataset.repeat(-1)
        test_dataset = test_dataset.take(steps_per_epoch)
        setattr(test_dataset, 'steps_per_epoch', steps_per_epoch)
        return test_dataset

    def get_config_info(self):
        return f'model-{self.network_cls.name}_' \
            f'mbs-{self.meta_batch_size}_' \
            f'n-{self.n}_' \
            f'k-{self.k}_' \
            f'stp-{self.num_steps_ml}'

    def create_meta_model(self, updated_model, model, gradients):
        k = 0
        variables = list()

        for i in range(len(model.layers)):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D) or \
                    isinstance(model.layers[i], tf.keras.layers.Dense):
                updated_model.layers[i].kernel = model.layers[i].kernel - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(updated_model.layers[i].kernel)
                updated_model.layers[i].bias = model.layers[i].bias - self.lr_inner_ml * gradients[k]
                k += 1
                variables.append(updated_model.layers[i].bias)

            elif isinstance(model.layers[i], tf.keras.layers.BatchNormalization):
                if hasattr(model.layers[i], 'moving_mean') and model.layers[i].moving_mean is not None:
                    updated_model.layers[i].moving_mean.assign(model.layers[i].moving_mean)
                if hasattr(model.layers[i], 'moving_variance') and model.layers[i].moving_variance is not None:
                    updated_model.layers[i].moving_variance.assign(model.layers[i].moving_variance)
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    updated_model.layers[i].beta = \
                        model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].beta)

            elif isinstance(model.layers[i], tf.keras.layers.LayerNormalization):
                if hasattr(model.layers[i], 'gamma') and model.layers[i].gamma is not None:
                    updated_model.layers[i].gamma = model.layers[i].gamma - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].gamma)
                if hasattr(model.layers[i], 'beta') and model.layers[i].beta is not None:
                    updated_model.layers[i].beta = \
                        model.layers[i].beta - self.lr_inner_ml * gradients[k]
                    k += 1
                    variables.append(updated_model.layers[i].beta)

        setattr(updated_model, 'meta_trainable_variables', variables)

    def get_train_loss_and_gradients(self, train_ds, train_labels):
        with tf.GradientTape(persistent=True) as train_tape:
            # TODO compare between model.forward(train_ds) and model(train_ds)
            logits = self.model(train_ds, training=True)
            train_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True))

        train_gradients = train_tape.gradient(train_loss, self.model.trainable_variables)
        return train_loss, train_gradients, train_tape

    def get_task_train_and_val_ds(self, task, labels):
        train_ds, val_ds = tf.split(task, num_or_size_splits=2)
        train_labels, val_labels = tf.split(labels, num_or_size_splits=2)

        train_ds = combine_first_two_axes(tf.squeeze(train_ds, axis=0))
        val_ds = combine_first_two_axes(tf.squeeze(val_ds, axis=0))
        train_labels = combine_first_two_axes(tf.squeeze(train_labels, axis=0))
        val_labels = combine_first_two_axes(tf.squeeze(val_labels, axis=0))

        return train_ds, val_ds, train_labels, val_labels

    def inner_train_loop(self, train_ds, train_labels, num_iterations=-1):
        if num_iterations == -1:
            num_iterations = self.num_steps_ml

            gradients = list()
            for variable in self.model.trainable_variables:
                gradients.append(tf.zeros_like(variable))

            self.create_meta_model(self.updated_models[0], self.model, gradients)

            for k in range(1, num_iterations + 1):
                with tf.GradientTape(persistent=True) as train_tape:
                    train_tape.watch(self.updated_models[k - 1].meta_trainable_variables)
                    logits = self.updated_models[k - 1](train_ds, training=True)
                    loss = tf.reduce_sum(
                        tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                    )
                gradients = train_tape.gradient(loss, self.updated_models[k - 1].meta_trainable_variables)
                self.create_meta_model(self.updated_models[k], self.updated_models[k - 1], gradients)

            return self.updated_models[-1]

        else:
            gradients = list()
            for variable in self.model.trainable_variables:
                gradients.append(tf.zeros_like(variable))

            self.create_meta_model(self.updated_models[0], self.model, gradients)
            copy_model = self.updated_models[0]

            for k in range(num_iterations):
                with tf.GradientTape(persistent=True) as train_tape:
                    train_tape.watch(copy_model.meta_trainable_variables)
                    logits = copy_model(train_ds, training=True)
                    loss = tf.reduce_sum(
                        tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True)
                    )
                gradients = train_tape.gradient(loss, copy_model.meta_trainable_variables)
                self.create_meta_model(copy_model, copy_model, gradients)

            return copy_model

    def save_model(self, epochs):
        # print("save model at ", os.path.join(self.checkpoint_dir, f'model_{epochs}_.ckpt'))
        self.model.save_weights(os.path.join(self.checkpoint_dir, f'model_{epochs}_.ckpt'))
        self.save_args(epochs)

    def save_args(self, epochs):
        arg_dict = {'self.n' :  self.n,
                    'self.k' :  self.k,
                    'self.meta_batch_size' :  self.meta_batch_size,
                    'self.num_steps_ml' :  self.num_steps_ml,
                    'self.lr_inner_ml' :  self.lr_inner_ml,
                    'self.num_steps_validation' :  self.num_steps_validation,
                    'self.save_after_epochs' :  self.save_after_epochs,
                    'self.log_train_images_after_iteration' :  self.log_train_images_after_iteration,
                    'self.report_validation_frequency' :  self.report_validation_frequency,
                    'self.meta_learning_rate' :  self.meta_learning_rate,
                    'self.least_number_of_tasks_val_test' :  self.least_number_of_tasks_val_test,
                    'self.clip_gradients' :  self.clip_gradients}
        with open(os.path.join(self.checkpoint_dir, f'model_arg_{epochs}_.bin'), 'wb') as f:
            pickle.dump(arg_dict, f)
        
    def load_model(self, epochs=None):
        epoch_count = 0
        if epochs is not None:
            try:
                print("Load model from : ", os.path.join(self.checkpoint_dir, f'model_arg_{epochs-1}_.bin'))
                with open(os.path.join(self.checkpoint_dir, f'model_arg_{epochs-1}_.bin'), 'rb') as f:
                    arg_dict_load = pickle.load(f)
                self.n = arg_dict_load['self.n']
                self.k = arg_dict_load['self.k']
                self.meta_batch_size = arg_dict_load['self.meta_batch_size']
                self.num_steps_ml = arg_dict_load['self.num_steps_ml']
                self.lr_inner_ml = arg_dict_load['self.lr_inner_ml']
                self.num_steps_validation = arg_dict_load['self.num_steps_validation']
                self.save_after_epochs = arg_dict_load['self.save_after_epochs']
                self.log_train_images_after_iteration = arg_dict_load['self.log_train_images_after_iteration']
                self.report_validation_frequency = arg_dict_load['self.report_validation_frequency']
                self.meta_learning_rate = arg_dict_load['self.meta_learning_rate']
                self.least_number_of_tasks_val_test = arg_dict_load['self.least_number_of_tasks_val_test']
                self.clip_gradients = arg_dict_load['self.clip_gradients']

                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_{epochs-1}_.ckpt')
                print("checkpoint_path : ", checkpoint_path)
                epoch_count = epochs
            except:
                print('not find checkpoint')
        else:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        if checkpoint_path is not None:
            try:
                self.model.load_weights(checkpoint_path)
                epoch_count = int(checkpoint_path.split("_")[-2])
                print('==================\nLoad Checkpoint\n(epoch : {})\n=================='.format(epoch_count))
            except Exception as e:
                print('Could not load the previous checkpoint!')

        else:
            print('No previous checkpoint found!')

        return epoch_count

    def meta_test(self, iterations = 5, epochs_to_load_from=None):
        self.test_dataset = self.get_test_dataset()
        epoch_count = self.load_model(epochs=epochs_to_load_from)

        test_accuracy_metric = tf.metrics.Accuracy()
        test_loss_metric = tf.metrics.Mean()

        for tmb, lmb in self.test_dataset:
            for task, labels in zip(tmb, lmb):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                updated_model = self.inner_train_loop(train_ds, train_labels, iterations)
                # If you want to compare with MAML paper, please set the training=True in the following line
                # In that paper the assumption is that we have access to all of test data together and we can evaluate
                # mean and variance from the batch which is given. Make sure to do the same thing in validation.
                updated_model_logits = updated_model(val_ds, training=True)

                self.update_loss_and_accuracy(updated_model_logits, val_labels, test_loss_metric, test_accuracy_metric)

            self.log_metric(self.test_summary_writer, 'Loss', test_loss_metric, step=1)
            self.log_metric(self.test_summary_writer, 'Accuracy', test_accuracy_metric, step=1)
    
        print('Test Loss: {}'.format(test_loss_metric.result().numpy()))
        print('Test Accuracy: {}'.format(test_accuracy_metric.result().numpy()))
        
        self.test_csv_path = os.path.join(self.test_log_dir, 'test.csv')
        if not os.path.isfile(self.test_csv_path):
            with open(self.test_csv_path, 'w', encoding='utf-8') as f:
                f.write("'test_count', 'accuracy', 'loss'\n")

        with open(self.test_csv_path, 'a', encoding='utf-8') as f:
            f.write("{}, {}, {}\n".format(self.test_count, test_accuracy_metric.result().numpy(), test_loss_metric.result().numpy()))
        self.test_count += 1
        
        # Save Adapted 
        step4_checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models')
        updated_model.save_weights(os.path.join(step4_checkpoint_dir, f'model_{epoch_count}_.ckpt'))
        arg_dict = {'self.n' :  self.n,
                    'self.k' :  self.k,
                    'self.meta_batch_size' :  self.meta_batch_size,
                    'self.num_steps_ml' :  self.num_steps_ml,
                    'self.lr_inner_ml' :  self.lr_inner_ml,
                    'self.num_steps_validation' :  self.num_steps_validation,
                    'self.save_after_epochs' :  self.save_after_epochs,
                    'self.log_train_images_after_iteration' :  self.log_train_images_after_iteration,
                    'self.report_validation_frequency' :  self.report_validation_frequency,
                    'self.meta_learning_rate' :  self.meta_learning_rate,
                    'self.least_number_of_tasks_val_test' :  self.least_number_of_tasks_val_test,
                    'self.clip_gradients' :  self.clip_gradients}
        with open(os.path.join(step4_checkpoint_dir, f'model_arg_{epoch_count}_.bin'), 'wb') as f:
            pickle.dump(arg_dict, f)
        

        return test_accuracy_metric.result().numpy()

    def log_images(self, summary_writer, train_ds, val_ds, step):
        with tf.device('cpu:0'):
            with summary_writer.as_default():
                tf.summary.image(
                    'train',
                    train_ds,
                    step=step,
                    max_outputs=5
                )
                tf.summary.image(
                    'validation',
                    val_ds,
                    step=step,
                    max_outputs=5
                )

    def update_loss_and_accuracy(self, logits, labels, loss_metric, accuracy_metric):
        # print(tf.argmax(logits, axis=-1))
        val_loss = tf.reduce_sum(
            tf.losses.categorical_crossentropy(labels, logits, from_logits=True))
        loss_metric.update_state(val_loss)
        accuracy_metric.update_state(
            tf.argmax(labels, axis=-1),
            tf.argmax(logits, axis=-1)
        )

    def log_metric(self, summary_writer, name, metric, step):
        with summary_writer.as_default():
            tf.summary.scalar(name, metric.result(), step=step)

    def report_validation_loss_and_accuracy(self, epoch_count):
        self.val_loss_metric.reset_states()
        self.val_accuracy_metric.reset_states()

        val_counter = 0
        for tmb, lmb in self.val_dataset:
            val_counter += 1
            for task, labels in zip(tmb, lmb):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                if val_counter % 5 == 0:
                    step = epoch_count * self.val_dataset.steps_per_epoch + val_counter
                    self.log_images(self.val_summary_writer, train_ds, val_ds, step)

                updated_model = self.inner_train_loop(train_ds, train_labels, self.num_steps_validation)
                # If you want to compare with MAML paper, please set the training=True in the following line
                # In that paper the assumption is that we have access to all of test data together and we can evaluate
                # mean and variance from the batch which is given. Make sure to do the same thing in evaluation.
                updated_model_logits = updated_model(val_ds, training=True)

                self.update_loss_and_accuracy(
                    updated_model_logits, val_labels, self.val_loss_metric, self.val_accuracy_metric
                )

        self.log_metric(self.val_summary_writer, 'Loss', self.val_loss_metric, step=epoch_count)
        self.log_metric(self.val_summary_writer, 'Accuracy', self.val_accuracy_metric, step=epoch_count)
        loss = self.val_loss_metric.result().numpy()
        acc = self.val_accuracy_metric.result().numpy()
        print('Validation Loss: {}'.format(loss))
        print('Validation Accuracy: {}'.format(acc))
                
        if self.skip_load_first_epoch_writing_val:
            # If it’s to continue training from the loaded model,
            # Need to skip writing first epoch's result 
            with open(self.val_csv_path, 'a', encoding='utf-8') as f:
                f.write("{}, {}, {}\n".format(epoch_count, acc, loss))
        else:
            self.skip_load_first_epoch_writing_val = True
            

    @tf.function
    def get_losses_of_tasks_batch(self, inputs):
        task, labels, iteration_count = inputs
        # print("task.shape : ", task.shape)
        # print("labels.shape : ", labels.shape)

        train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
        # print("train_ds.shape : ", train_ds.shape)
        # print("val_ds.shape : ", val_ds.shape)
        # print("train_labels.shape : ", train_labels.shape)
        # print("val_labels.shape : ", val_labels.shape)
        # exit()

        if self.log_train_images_after_iteration != -1 and \
                iteration_count % self.log_train_images_after_iteration == 0:

            self.log_images(self.train_summary_writer, train_ds, val_ds, step=iteration_count)

            with tf.device('cpu:0'):
                with self.train_summary_writer.as_default():
                    for var in self.model.variables:
                        tf.summary.histogram(var.name, var, step=iteration_count)

                    for k in range(len(self.updated_models)):
                        var_count = 0
                        if hasattr(self.updated_models[k], 'meta_trainable_variables'):
                            for var in self.updated_models[k].meta_trainable_variables:
                                var_count += 1
                                tf.summary.histogram(f'updated_model_{k}_' + str(var_count), var, step=iteration_count)

        updated_model = self.inner_train_loop(train_ds, train_labels, -1)
        updated_model_logits = updated_model(val_ds, training=True)
        val_loss = tf.reduce_sum(
            tf.losses.categorical_crossentropy(val_labels, updated_model_logits, from_logits=True)
        )
        self.train_loss_metric.update_state(val_loss)
        self.train_accuracy_metric.update_state(
            tf.argmax(val_labels, axis=-1),
            tf.argmax(updated_model_logits, axis=-1)
        )
        return val_loss

    def meta_train_loop(self, tasks_meta_batch, labels_meta_batch, iteration_count):
        with tf.GradientTape(persistent=True) as outer_tape:
            tasks_final_losses = tf.map_fn(
                self.get_losses_of_tasks_batch,
                elems=(
                    tasks_meta_batch,
                    labels_meta_batch,
                    tf.cast(tf.ones(self.meta_batch_size, 1) * iteration_count, tf.int64)
                ),
                dtype=tf.float32,
                parallel_iterations=self.meta_batch_size
            )
            final_loss = tf.reduce_mean(tasks_final_losses)
        outer_gradients = outer_tape.gradient(final_loss, self.model.trainable_variables)
        if self.clip_gradients:
            outer_gradients = [tf.clip_by_value(grad, -10, 10) for grad in outer_gradients]
        self.optimizer.apply_gradients(zip(outer_gradients, self.model.trainable_variables))

    def meta_train(self, epochs=4000):
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        start_epoch = self.load_model()
        iteration_count = start_epoch * self.train_dataset.steps_per_epoch
        print("-"*100)
        print(start_epoch)
        if start_epoch != 0:
            self.skip_load_first_epoch_writing_train = False
            self.skip_load_first_epoch_writing_val = False
        else:
            self.skip_load_first_epoch_writing_train = True
            self.skip_load_first_epoch_writing_val = True

        pbar = tqdm(self.train_dataset)
        
        self.train_csv_path = os.path.join(self.train_log_dir, 'train.csv')
        if not os.path.isfile(self.train_csv_path):
            with open(self.train_csv_path, 'w', encoding='utf-8') as f:
                f.write("'epoch', 'accuracy', 'loss'\n")

        self.val_csv_path = os.path.join(self.val_log_dir, 'val.csv')
        if not os.path.isfile(self.val_csv_path):
            with open(self.val_csv_path, 'w', encoding='utf-8') as f:
                f.write("'epoch', 'accuracy', 'loss'\n")

        for epoch_count in range(start_epoch, epochs):
            if epoch_count != 0:
                # Save val metrics per report_validation_frequency
                if epoch_count % self.report_validation_frequency == 0:
                    self.report_validation_loss_and_accuracy(epoch_count)
                    loss = self.train_loss_metric.result().numpy()
                    acc = self.train_accuracy_metric.result().numpy()
                    print('Train Loss: {}'.format(loss))
                    print('Train Accuracy: {}'.format(acc))
                    
                # Save training metrics
                loss = self.train_loss_metric.result().numpy()
                acc = self.train_accuracy_metric.result().numpy()
                
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Loss', loss, step=epoch_count)
                    tf.summary.scalar('Accuracy', acc, step=epoch_count)
                
                if self.skip_load_first_epoch_writing_train:
                    print("Write")
                    # If it’s to continue training from the loaded model,
                    # Need to skip writing first epoch's result 
                    with open(self.train_csv_path, 'a', encoding='utf-8') as f:
                        f.write("{}, {}, {}\n".format(epoch_count, acc, loss))
                else:
                    print("Pass")
                    self.skip_load_first_epoch_writing_train = True

                    
                if epoch_count % self.save_after_epochs == 0:
                    self.save_model(epoch_count)

            self.train_accuracy_metric.reset_states()
            self.train_loss_metric.reset_states()

            for tasks_meta_batch, labels_meta_batch in self.train_dataset:
                # print("tasks_meta_batch.shape : ", tasks_meta_batch.shape)
                # print("labels_meta_batch.shape : ", tasks_meta_batch.shape)
                self.meta_train_loop(tasks_meta_batch, labels_meta_batch, iteration_count)
                iteration_count += 1
                pbar.set_description_str('Epoch{}, Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                    epoch_count,
                    iteration_count,
                    self.train_loss_metric.result().numpy(),
                    self.train_accuracy_metric.result().numpy()
                ))
                pbar.update(1)
    # 20.10.14. Change the method name : predict_with_support -> meta_predict
    def predict_with_support(self, save_path, meta_test_path, iterations = 5, epochs_to_load_from=None):
        meta_test_path = os.path.join(os.getcwd(), meta_test_path)
        dataset_folders = [
            os.path.join(meta_test_path, class_name) for class_name in os.listdir(meta_test_path)
        ]


        predict_dataset = self.database.get_supervised_meta_learning_dataset(
            dataset_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
        )
        self.load_model(epochs=epochs_to_load_from)
        # Load whole test dataset and predict 
        steps_per_epoch = max(predict_dataset.steps_per_epoch, self.least_number_of_tasks_val_test)
        test_dataset = predict_dataset.repeat(-1)
        test_dataset = test_dataset.take(steps_per_epoch)
        save_path_base = os.path.join(save_path, 'output')
        os.makedirs(save_path_base, exist_ok=True)

        for tmb, lmb in test_dataset:
            for idx, (task, labels) in enumerate(zip(tmb, lmb)):
                train_ds, val_ds, train_labels, val_labels = self.get_task_train_and_val_ds(task, labels)
                updated_model = self.inner_train_loop(train_ds, train_labels, iterations)
                # If you want to compare with MAML paper, please set the training=True in the following line
                # In that paper the assumption is that we have access to all of test data together and we can evaluate
                # mean and variance from the batch which is given. Make sure to do the same thing in validation.

                support_targets_ = tf.argmax(train_labels, axis=-1)
                val_targets_ = tf.argmax(val_labels, axis=-1)

                updated_model_logits = updated_model(val_ds, training=False)
                result = tf.argmax(updated_model_logits, axis=-1)

                task_name = "task{}".format(idx)
                result_ = result.numpy()

                for img, label in zip(train_ds, support_targets_):
                    save_path_ = os.path.join(save_path_base, "[support_set]{}_{}.png".format(task_name, label))
                    tf.keras.preprocessing.image.save_img(save_path_, img, data_format='channels_last', file_format=None, scale=True)

                for img, label, predict in zip(val_ds, val_targets_, result_):
                    save_path_ = os.path.join(save_path_base, "[query_set]{}_{}_{}.png".format(task_name, label, predict))
                    tf.keras.preprocessing.image.save_img(save_path_, img, data_format='channels_last', file_format=None, scale=True)

                break
            break

        # Saving N-way K-shot JSON
        from utils import save_nwaykshot

        self.database.is_preview = True

        predict_dataset = self.database.get_supervised_meta_learning_dataset(
            dataset_folders,
            n=self.n,
            k=self.k,
            meta_batch_size=1,
        )

        # Numbering the classees
        folders = sorted(dataset_folders)
        
        class2num = { i.split(os.sep)[-1]: 'class{}'.format(n) for n, i in enumerate(folders) }
        print(class2num)
        num2class = {v : k for k, v in class2num.items()}
        
        # Save the N-way K-shot task json file (for tarin set)
        json_save_path = os.path.join(self._root, 'nwaykshot_{}.json'.format(self.args.benchmark_dataset))
        save_nwaykshot(predict_dataset, json_save_path, class2num)

        return result