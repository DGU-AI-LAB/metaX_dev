from model.optimization_based.maml import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearning
from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse

import logging, os
import pickle
from configparser import ConfigParser

config = ConfigParser()

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':

    # 빠른 테스트를 위한 세팅
    parser = argparse.ArgumentParser()
    #parser.add_argument('--benchmark_dataset', type=str, default='mini_imagenet') # 20.09.03
    #parser.add_argument('--network_cls', type=str, default='mini_imagenet')       # 20.09.03
    parser.add_argument('--benchmark_dataset', type=str, default='omniglot')       # 20.09.03
    parser.add_argument('--network_cls', type=str, default='omniglot')             # 20.09.03
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=7) # 5
    parser.add_argument('--iterations', type=int, default=1) # 5
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=2)  # 20.09.03
    parser.add_argument('--num_steps_ml', type=int, default=1) # 10
    parser.add_argument('--lr_inner_ml', type=float, default=0.4)
    parser.add_argument('--num_steps_validation', type=int, default=1) # 10
    # parser.add_argument('--save_after_epochs', type=int, default=500)
    parser.add_argument('--save_after_epochs', type=int, default=1)
    parser.add_argument('--meta_learning_rate', type=float, default=0.001)
    parser.add_argument('--report_validation_frequency', type=int, default=1)
    parser.add_argument('--log_train_images_after_iteration', type=int, default=1)


    # # 학습에 필요한 인자를 입력으로 받습니다.
    # # 아래 인자들은 메타러닝 세팅에 대한 것으로 일반 학습에 대한 세팅은 다를 수 있습니다.
    # parser = argparse.ArgumentParser()

    # # Argument for Common Deep Learning
    # # It take user input & It also have default values
    # parser.add_argument('--benchmark_dataset', type=str, default='omniglot')
    # parser.add_argument('--network_cls', type=str, default='omniglot')
    # parser.add_argument('--epochs', type=int, default=5)
    # parser.add_argument('--meta_learning_rate', type=float, default=0.001) # Corresponds to learning rate in general deep learning

    # # Argument for Meta-Learning
    # # It take user input & It also have default values
    # parser.add_argument('--n', type=int, default=5) 
    # parser.add_argument('--k', type=int, default=1)
    # parser.add_argument('--meta_batch_size', type=int, default=5)
    # parser.add_argument('--num_steps_ml', type=int, default=10)
    # parser.add_argument('--lr_inner_ml', type=float, default=0.4)
    # parser.add_argument('--iterations', type=int, default=5)
    # parser.add_argument('--num_steps_validation', type=int, default=10)
    
    # # Argument for wrting log & save file
    # # It take user input & It also have default values
    # parser.add_argument('--save_after_epochs', type=int, default=1)
    # parser.add_argument('--report_validation_frequency', type=int, default=50)
    # parser.add_argument('--log_train_images_after_iteration', type=int, default=1)

    args = parser.parse_args()


    # Create config file
    config['common_DL'] = {
        'benchmark_dataset' : args.benchmark_dataset,
        'network_cls' : args.network_cls,
        'epochs' : args.epochs,
        'meta_learning_rate' : args.meta_learning_rate,
    }

    config['MetaLearning'] = {
        'n' : args.n,
        'k' : args.k,
        'meta_batch_size' : args.meta_batch_size,
        'num_steps_ml' : args.num_steps_ml,
        'lr_inner_ml' : args.lr_inner_ml,
        'iterations' : args.iterations,
        'num_steps_validation' : args.num_steps_validation,
    }

    config['LogSave'] = {
        'save_after_epochs' : args.save_after_epochs,
        'report_validation_frequency' : args.report_validation_frequency,
        'log_train_images_after_iteration' : args.log_train_images_after_iteration,
    }


    # Check & Load the existance of *.pkl database file
    base_path_step1 = os.path.join(os.getcwd(),
     'dataset/data/ui_output','maml_{}'.format(args.benchmark_dataset), 'step1')
    os.makedirs(base_path_step1, exist_ok=True)
    save_path_step1 = os.path.join(base_path_step1, '{}.pkl'.format(args.benchmark_dataset))

    base_path = os.path.join(os.getcwd(),
     'dataset/data/ui_output','maml_{}'.format(args.benchmark_dataset), 'step3')
    os.makedirs(base_path, exist_ok=True)

    def _get_config_info(args):
        return f'model-{args.network_cls}_' \
            f'mbs-{args.meta_batch_size}_' \
            f'n-{args.n}_' \
            f'k-{args.k}_' \
            f'stp-{args.num_steps_ml}'
    
    # Save Step3 argments
    step3_args_base_path = os.path.join(base_path, 'args') 
    os.makedirs(step3_args_base_path, exist_ok=True)
    step3_args_path = os.path.join(base_path, 'args', 'args_{}.ini'.format(_get_config_info(args)))
    with open(step3_args_path, 'w') as f:
        config.write(f)
    print("Step3 args are saved")

    if os.path.isfile(save_path_step1):
        print("Load dataset")
        with open(save_path_step1, 'rb') as f:
            database = pickle.load(f)
    else:
        # 데이터셋 객체를 생성합니다.
        # 타입 : tf.data.Dataset
        if args.benchmark_dataset == "omniglot":
            database = OmniglotDatabase(
                # 200831 changed path, add raw_data folder
                raw_data_address="dataset/raw_data/omniglot",
                random_seed=47,
                num_train_classes=1200,
                num_val_classes=100)

        elif args.benchmark_dataset == "mini_imagenet":
            database=MiniImagenetDatabase(
                # 200831 changed path, add raw_data folder
                raw_data_address="\dataset\\raw_data\\mini_imagenet",
                random_seed=-1)

        # Save the database file
        with open(save_path_step1, 'wb') as f:
            pickle.dump(database, f) # e.g. for omniglot, ./dataset/data/ui_output/maml/step1/omniglot.pkl
        # -> To laod this file in the next step

    # 모델 객체를 생성합니다.
    if args.network_cls == "omniglot":
        network_cls=OmniglotModel
    elif args.network_cls == "mini_imagenet":
        network_cls=MiniImagenetModel

    # 학습을 위한 클래스를 생성합니다.
    maml = ModelAgnosticMetaLearning(args, database, network_cls)
    # args : 파라미터      type : parser.parse_args
    # database : 데이터셋  type : database
    # network_cls : 모델   type : MetaLearning
    
    # Set the save path of the log files and the model weights
    maml._root = base_path
    
    # 학습을 위한 클래스를 사용하여 입력받은 파라미터를 통해 meta_train을 수행합니다.
    maml.meta_train(epochs = args.epochs)
    # epochs : 반복 횟수 type : int