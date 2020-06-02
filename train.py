from model.optimization_based.MAML import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearningModel
from dataset.data_generator_MAML import OmniglotDatabase, MiniImagenetDatabase
import argparse
from glob import glob
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='mini_imagenet')
    parser.add_argument('--network_cls', type=str, default='mini_imagenet')
    # parser.add_argument('--database', type=str, default='omniglot')
    # parser.add_argument('--network_cls', type=str, default='omniglot')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=32)
    parser.add_argument('--num_steps_ml', type=int, default=10)
    parser.add_argument('--lr_inner_ml', type=float, default=0.4)
    parser.add_argument('--num_steps_validation', type=int, default=10)
    parser.add_argument('--save_after_epochs', type=int, default=500)
    parser.add_argument('--meta_learning_rate', type=float, default=0.001)
    parser.add_argument('--report_validation_frequency', type=int, default=50)
    parser.add_argument('--log_train_images_after_iteration', type=int, default=1)

    args = parser.parse_args()
    
    if args.database == "omniglot":
        database = OmniglotDatabase(
            raw_data_address="dataset\omniglot",
            random_seed=47,
            num_train_classes=1200,
            num_val_classes=100)
    elif args.database == "mini_imagenet":
        database=MiniImagenetDatabase(
            raw_data_address="\dataset\mini_imagenet",
            random_seed=-1)
    if args.network_cls == "omniglot":
        network_cls=OmniglotModel
    elif args.network_cls == "mini_imagenet":
        network_cls=MiniImagenetModel

    # train_dict, val_dict, test_dict = database.get_class()
    # print(train_dict.keys())
    # print(database.preview_image(train_dict['Korean_character40'][0]))
    # print(database.preview_image(train_dict['n01532829'][0]))


    maml = ModelAgnosticMetaLearningModel(args, database, network_cls)

    # maml.train(epochs = args.epochs)
    # maml.evaluate(iterations = args.iterations)
    print(maml.predict())
