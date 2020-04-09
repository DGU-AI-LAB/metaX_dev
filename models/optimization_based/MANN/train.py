from MANN import MANN

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
    flags.DEFINE_integer('num_samples', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
    flags.DEFINE_integer('meta_batch_size', 4, 'Number of N-way classification tasks per batch')
    flags.DEFINE_integer('training_step', 2001, 'Total training step')
    flags.DEFINE_integer('visualization_step', 100, 'Visualizatino step')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
    flags.DEFINE_boolean('shuffle', True, 'T/F for episode of meta learnig dataset')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[1], 'GPU') # use second GPU
                tf.config.experimental.set_memory_growth(gpus[1], True)
            except RuntimeError as error_message:
                print(error_message)
                
    data_generator = DataGenerator(FLAGS.num_classes, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_samples + 1)
    model = MANN(num_classes=FLAGS.num_classes, samples_per_class=FLAGS.num_samples + 1)

    model.train(FLAGS, data_generator)
            