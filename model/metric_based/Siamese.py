import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
import pickle
from tqdm import tqdm
from utils import combine_first_two_axes, createFolder
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization, MaxPool2D
from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
from model.LearningType import MetaLearning

class OmniglotModel(tf.keras.Model):
    name = 'OmniglotModel'
    def __init__(self, num_classes = 2):
        super(OmniglotModel, self).__init__(name='omniglot_model')
        self.conv1 = Conv2D(64, 10, name='conv1', strides=(1, 1), activation = 'relu',)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = Conv2D(128, 7, name='conv2', strides=(1, 1), activation = 'relu',)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = Conv2D(128, 4, name='conv3', strides=(1, 1), activation = 'relu',)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = Conv2D(256, 4, name='conv4',  strides=(1, 1), activation = 'relu',) 
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        
        self.maxpool = MaxPool2D((2, 2), name="pooling")
        self.flatten = Flatten(name='flatten')        
        self.dense = Dense(128, activation='sigmoid', name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        return tf.keras.activations.relu(batch_normalized_out)

    def call_embed(self, x, training=False):
        conv1_output = self.conv_block(x, self.conv1, self.bn1, training)
        conv2_output = self.conv_block(conv1_output, self.conv2, self.bn2, training)
        conv3_output = self.conv_block(conv2_output, self.conv3, self.bn3, training)
        conv4_output = self.conv_block(conv3_output, self.conv4, self.bn4, training)
        flatten_output = self.flatten(conv4_output)
        output = self.dense(flatten_output)        
        return output
        
    def call(self, x0, x1, anc, training=False):        
        x0_embed = self.call_embed(x0, training)
        x1_embed = self.call_embed(x1, training)      
        anc_embed = self.call_embed(anc, training)
        
        x0_dist = tf.norm(tf.abs(tf.subtract(anc_embed, x0_embed)), axis = -1)
        x1_dist = tf.norm(tf.abs(tf.subtract(anc_embed, x1_embed)), axis = -1)
        result = tf.math.argmin(tf.stack([x0_dist, x1_dist], axis = -1), axis = -1)
        result_onehot = tf.one_hot(result, depth = 2)        
        return x0_dist, x1_dist, result_onehot

class MiniImagenetModel(tf.keras.Model):
    name = 'MiniImagenetModel'
    def __init__(self, num_classes = 2):
        super(OmniglotModel, self).__init__(name='omniglot_model')
        self.conv1 = Conv2D(64, 10, name='conv1', strides=(1, 1), activation = 'relu',)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn1')
        self.conv2 = Conv2D(128, 7, name='conv2', strides=(1, 1), activation = 'relu',)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn2')
        self.conv3 = Conv2D(128, 4, name='conv3', strides=(1, 1), activation = 'relu',)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn3')
        self.conv4 = Conv2D(256, 4, name='conv4',  strides=(1, 1), activation = 'relu',) 
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0, center=True, scale=False, name='bn4')
        
        self.maxpool = MaxPool2D((2, 2), name="pooling")
        self.flatten = Flatten(name='flatten')        
        self.dense = Dense(128, activation='sigmoid', name='dense')

    def conv_block(self, features, conv, bn=None, training=False):
        conv_out = conv(features)
        batch_normalized_out = bn(conv_out, training=training)
        return tf.keras.activations.relu(batch_normalized_out)

    def call_embed(self, x, training=False):
        conv1_output = self.conv_block(x, self.conv1, self.bn1, training)
        conv2_output = self.conv_block(conv1_output, self.conv2, self.bn2, training)
        conv3_output = self.conv_block(conv2_output, self.conv3, self.bn3, training)
        conv4_output = self.conv_block(conv3_output, self.conv4, self.bn4, training)
        flatten_output = self.flatten(conv4_output)
        output = self.dense(flatten_output)        
        return output
        
    def call(self, x0, x1, anc, training=False):        
        x0_embed = self.call_embed(x0, training)
        x1_embed = self.call_embed(x1, training)      
        anc_embed = self.call_embed(anc, training)
        
        x0_dist = tf.norm(tf.abs(tf.subtract(anc_embed, x0_embed)), axis = -1)
        x1_dist = tf.norm(tf.abs(tf.subtract(anc_embed, x1_embed)), axis = -1)
        result = tf.math.argmin(tf.stack([x0_dist, x1_dist], axis = -1), axis = -1)
        result_onehot = tf.one_hot(result, depth = 2)        
        return x0_dist, x1_dist, result_onehot

class Siamese(MetaLearning):
    def __init__(
            self,
            args,
            database,
            network_cls,
            clip_gradients=False
        ):
        # Common Meta-Learning hyperparameters
        super(Siamese, self).__init__(
                        args,
                        database,
                        network_cls,
                        clip_gradients=False
        )


        # Initialize the MAML
        self.model(tf.zeros(shape=(1, *self.database.input_shape)), tf.zeros(shape=(1, *self.database.input_shape)), tf.zeros(shape=(1, *self.database.input_shape)))
        self.margin = args.margin
        
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

        self._root = self.get_root()                                                         # type : string
        self.train_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/train/') # type : string

        try:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        except:
            createFolder(self.train_log_dir)
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/val/')
        try:
            self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        except:
            createFolder(self.val_log_dir)
            self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models') # 20.09.03

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
            f'margin-{self.margin}'

    def get_task_train_and_val_ds(self, task, labels):
        train_ds, val_ds = tf.split(task, num_or_size_splits=2)
        train_labels, val_labels = tf.split(labels, num_or_size_splits=2)

        train_ds = combine_first_two_axes(tf.squeeze(train_ds, axis=0))
        val_ds = combine_first_two_axes(tf.squeeze(val_ds, axis=0))
        train_labels = combine_first_two_axes(tf.squeeze(train_labels, axis=0))
        val_labels = combine_first_two_axes(tf.squeeze(val_labels, axis=0))

        return train_ds, val_ds, train_labels, val_labels

    def save_model(self, epochs):
        # print("save model at ", os.path.join(self.checkpoint_dir, f'model_{epochs}_.ckpt'))
        self.model.save_weights(os.path.join(self.checkpoint_dir, f'model_{epochs}_.ckpt'))
        self.save_args(epochs)

    def save_args(self, epochs):
        arg_dict = {'self.n' :  self.n,
                    'self.k' :  self.k,
                    'self.meta_batch_size' :  self.meta_batch_size,
                    # 'self.num_steps_ml' :  self.num_steps_ml,
                    # 'self.lr_inner_ml' :  self.lr_inner_ml,
                    # 'self.num_steps_validation' :  self.num_steps_validation,
                    'self.save_after_epochs' :  self.save_after_epochs,
                    'self.log_train_images_after_iteration' :  self.log_train_images_after_iteration,
                    'self.report_validation_frequency' :  self.report_validation_frequency,
                    'self.meta_learning_rate' :  self.meta_learning_rate,
                    # 'self.clip_gradients' :  self.clip_gradients,
                    'self.least_number_of_tasks_val_test' :  self.least_number_of_tasks_val_test}
        with open(os.path.join(self.checkpoint_dir, f'model_arg_{epochs}_.bin'), 'wb') as f:
            pickle.dump(arg_dict, f)
        
    def load_model(self, epochs=None):
        epoch_count = 0
        if epochs is not None:
            try:
                with open(os.path.join(self.checkpoint_dir, f'model_arg_{epochs-1}_.bin'), 'rb') as f:
                    arg_dict_load = pickle.load(f)
                self.n = arg_dict_load['self.n']
                self.k = arg_dict_load['self.k']
                self.meta_batch_size = arg_dict_load['self.meta_batch_size']
                # self.num_steps_ml = arg_dict_load['self.num_steps_ml']
                # self.lr_inner_ml = arg_dict_load['self.lr_inner_ml']
                # self.num_steps_validation = arg_dict_load['self.num_steps_validation']
                self.save_after_epochs = arg_dict_load['self.save_after_epochs']
                self.log_train_images_after_iteration = arg_dict_load['self.log_train_images_after_iteration']
                self.report_validation_frequency = arg_dict_load['self.report_validation_frequency']
                self.meta_learning_rate = arg_dict_load['self.meta_learning_rate']
                # self.least_number_of_tasks_val_test = arg_dict_load['self.least_number_of_tasks_val_test']
                # self.clip_gradients = arg_dict_load['self.clip_gradients']

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
        return test_accuracy_metric.result().numpy()

    def log_metric(self, summary_writer, name, metric, step):
        with summary_writer.as_default():
            tf.summary.scalar(name, metric.result(), step=step)

    def triplet_loss(self, x0_dist, x1_dist, label):
        temp = tf.constant(range(label.shape[0]), tf.int64)
        pos_index = tf.expand_dims(tf.stack([temp, tf.argmax(label, axis = -1)], axis = 1), 0)  
        neg_index = tf.expand_dims(tf.stack([temp, tf.argmin(label, axis = -1)], axis = 1), 0)        
        
        dist = tf.expand_dims(tf.stack([x0_dist, x1_dist], axis = 1), 0)
        
        pos = tf.gather_nd(dist, pos_index, batch_dims = 1)
        neg = tf.gather_nd(dist, neg_index, batch_dims = 1)
        
        loss = tf.reduce_mean(tf.maximum(pos - neg + self.margin, 0))
        return loss
    
    def update_loss_and_accuracy(self, logits, labels, val_loss, loss_metric, accuracy_metric):
        # print(tf.argmax(logits, axis=-1))
        loss_metric.update_state(val_loss)
        accuracy_metric.update_state(
            tf.argmax(labels, axis=-1),
            tf.argmax(logits, axis=-1)
        )

    # loss 함수 수정
    # inner loop 구문 제거, 학습 알고리즘 재 작성
    def meta_test(self, epochs_to_load_from=None):
        self.test_dataset = self.get_test_dataset()
        self.load_model(epochs=epochs_to_load_from)
        test_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/test/')
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        test_accuracy_metric = tf.metrics.Accuracy()
        test_loss_metric = tf.metrics.Mean()

        for tmb, lmb in self.test_dataset:
            for task, labels in zip(tmb, lmb):
                # (n, image_shape) (n, image_shape) (n, n) (n, n)
                support, query, support_labels, query_labels = self.get_task_train_and_val_ds(task, labels)
                
                s0, s1 = tf.split(support, 2, axis = 0)
                q, _ = tf.split(query, 2, axis = 0)
                y, _ = tf.split(query_labels, 2, axis = 0)
                s0_dist, s1_dist, y_pred = self.model(s0, s1, q)
                test_loss = self.triplet_loss(s0_dist, s1_dist, y)
                self.update_loss_and_accuracy(y_pred, y, test_loss, test_loss_metric, test_accuracy_metric)

            self.log_metric(test_summary_writer, 'Loss', test_loss_metric, step=1)
            self.log_metric(test_summary_writer, 'Accuracy', test_accuracy_metric, step=1)
        
        print()
        print('Test Loss: {}'.format(test_loss_metric.result().numpy()))
        print('Test Accuracy: {}'.format(test_accuracy_metric.result().numpy()))

    # batch, 2(support&query), n, k, image(h,w,c) ==> batch, 2, 2, 1, image
    # batch, 2(support&query), n, k, n(onehot)    ==> batch, 2, 2, 1, 2
    def get_train_loss_and_gradients(self, train_ds, train_labels):
        with tf.GradientTape(persistent=True) as train_tape:
            # TODO compare between model.forward(train_ds) and model(train_ds)
            support, query = tf.split(train_ds, 2, axis = 1)
            _, y = tf.split(train_labels, 2, axis = 1)   
            
            s0, s1 = tf.split(support, 2, axis = 2)
            s0 = tf.squeeze(s0, [1, 2, 3]) # (batch, image_shape)
            s1 = tf.squeeze(s1, [1, 2, 3]) # (batch, image_shape)
            
            q, _ = tf.split(query, 2, axis = 2)        
            q = tf.squeeze(q, [1, 2, 3]) # (batch, image_shape)             
                 
            y, _ = tf.split(y, 2, axis = 2)        
            y = tf.squeeze(y, [1, 2, 3]) # (batch, 2)
            
            s0_dist, s1_dist, y_pred = self.model(s0, s1, q, training=True) # (batch,) (batch,) (batch, 2)
            
            # train_loss = tf.reduce_sum(tf.losses.categorical_crossentropy(train_labels, logits, from_logits=True))
            train_loss = self.triplet_loss(s0_dist, s1_dist, y)
        
        self.train_loss_metric.update_state(train_loss)
        self.train_accuracy_metric.update_state(
            tf.argmax(y, 1),
            tf.argmax(y_pred, 1)
        )
        train_gradients = train_tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(train_gradients, self.model.trainable_variables))
        return train_loss, train_gradients, train_tape
        

    # loss 함수 수정
    # 학습법 수정
    def meta_train(self, epochs=40):
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        start_epoch = self.load_model()
        iteration_count = start_epoch * self.train_dataset.steps_per_epoch

        pbar = tqdm(self.train_dataset)
        
        for epoch_count in range(start_epoch, epochs):
            if epoch_count != 0:
                if epoch_count % self.report_validation_frequency == 0:
                    self.report_validation_loss_and_accuracy(epoch_count)
                    if epoch_count != 0:
                        print('Train Loss: {}'.format(self.train_loss_metric.result().numpy()))
                        print('Train Accuracy: {}'.format(self.train_accuracy_metric.result().numpy()))
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Loss', self.train_loss_metric.result(), step=epoch_count)
                    tf.summary.scalar('Accuracy', self.train_accuracy_metric.result(), step=epoch_count)
    
                if epoch_count % self.save_after_epochs == 0:
                    self.save_model(epoch_count)

            self.train_accuracy_metric.reset_states()
            self.train_loss_metric.reset_states()
            
            for tasks_meta_batch, labels_meta_batch in self.train_dataset:
                # print("tasks_meta_batch.shape : ", tasks_meta_batch.shape)  # batch, 2(support&query), n, k, image(h,w,c) ==> batch, 2, 2, 1, image
                # print("labels_meta_batch.shape : ", labels_meta_batch.shape)# batch, 2(support&query), n, k, n(onehot)    ==> batch, 2, 2, 1, 2
                
                train_loss, train_gradients, train_tape = self.get_train_loss_and_gradients(tasks_meta_batch, labels_meta_batch)

                iteration_count += 1
                pbar.set_description_str('Epoch{}, Iteration{}: Train Loss: {}, Train Accuracy: {}'.format(
                    epoch_count,
                    iteration_count,
                    self.train_loss_metric.result().numpy(),
                    self.train_accuracy_metric.result().numpy()
                ))
                pbar.update(1)

    # predict 수정
    def predict_with_support(self, meta_test_path, epochs_to_load_from=None):
        meta_test_path = os.getcwd() + meta_test_path
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
        for tmb, lmb in test_dataset:
            for task, labels in zip(tmb, lmb):
                support, query, support_labels, query_labels = self.get_task_train_and_val_ds(task, labels)
                
                s0, s1 = tf.split(support, 2, axis = 0)
                q, _ = tf.split(query, 2, axis = 0)
                y, _ = tf.split(query_labels, 2, axis = 0)
                s0_dist, s1_dist, y_pred = self.model(s0, s1, q)

                result = tf.argmax(y_pred, axis=-1)
                break
            break
        return result