from model.optimization_based.maml import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearning
from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse

import logging, os
import pickle
import glob
from configparser import ConfigParser

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
      # User Input Argument
    # - num_steps_ml : The number of inner steps 
    # - iterations : Inner loop의 gradient update 횟수
    # - num_steps_validation : validation set에 대한  fine tuning 스텝수

    parser = argparse.ArgumentParser()


    # Config File Load : Step 2 Config file
    config_parser = ConfigParser()
    maml_path = os.path.join(os.getcwd(), 'dataset/data/ui_output'.replace('/', os.sep),'maml')
    args_path = os.path.join(maml_path, 'args') 
    step3_args_path = os.path.join(args_path, 'step2.ini')
    config_parser.read(step3_args_path)
    print("Load Step3 arguments from : {}".format(step3_args_path))



    # Load latest step4_{config}.ini
    config_parser = ConfigParser()
    maml_path = os.path.join(os.getcwd(), 'dataset/data/ui_output'.replace('/', os.sep),'maml')
    args_path = os.path.join(maml_path, 'args', '*') 
    list_of_args_ini_files = glob.glob(args_path)
    print(list_of_args_ini_files)
    list_of_args_ini_files = [i for i in list_of_args_ini_files if 'step4_' in i]
    latest_file = max(list_of_args_ini_files, key=os.path.getctime)
    print("Load Step4 arguments from : {}".format(latest_file))
    config_parser.read(latest_file)


    # 학습에 필요한 인자를 입력으로 받습니다.
    # 아래 인자들은 메타러닝 세팅에 대한 것으로 일반 학습에 대한 세팅은 다를 수 있습니다.
    parser = argparse.ArgumentParser()

    # User Input of STEP4 
    # The default values are the values of step3
    # 1. num_steps_ml         : int : The number of inner steps 
    # 2. iterations           : int : inner loop의 gradient update 횟수
    # 3. num_steps_validation : int : validation set에 대한  fine tuning 스텝수
    # Othre arguments are loaded from args_###.ini file that has saved in the step3
 
    # Argument for Common Deep Learning
    # It take user input & It also have default values
    parser.add_argument('--benchmark_dataset', type=str, default=config_parser['common_DL']['benchmark_dataset'])
    parser.add_argument('--network_cls', type=str, default=config_parser['common_DL']['benchmark_dataset'])
    parser.add_argument('--epochs', type=int, default=config_parser['common_DL']['epochs'])
    parser.add_argument('--meta_learning_rate', type=float, default=config_parser['common_DL']['meta_learning_rate'])

    # Argument for Meta-Learning
    # It take user input & It also have default values
    parser.add_argument('--n', type=int, default=config_parser['MetaLearning']['n']) 
    parser.add_argument('--k', type=int, default=config_parser['MetaLearning']['k'])
    parser.add_argument('--meta_batch_size', type=int, default=config_parser['MetaLearning']['meta_batch_size'])
    parser.add_argument('--num_steps_ml', type=int, default=config_parser['MetaLearning']['num_steps_ml'])                 # User Input of STEP4
    parser.add_argument('--lr_inner_ml', type=float, default=config_parser['MetaLearning']['lr_inner_ml'])
    parser.add_argument('--iterations', type=int, default=config_parser['MetaLearning']['iterations'])     # User Input of STEP4
    parser.add_argument('--num_steps_validation', type=int, default=config_parser['MetaLearning']['num_steps_validation']) # User Input of STEP4
    
    # Argument for wrting log & save file
    # It take user input & It also have default values
    parser.add_argument('--save_after_epochs', type=int, default=config_parser['LogSave']['save_after_epochs'])
    parser.add_argument('--report_validation_frequency', type=int, default=config_parser['LogSave']['report_validation_frequency'])
    parser.add_argument('--log_train_images_after_iteration', type=int, default=config_parser['LogSave']['log_train_images_after_iteration'])

    args = parser.parse_args()


    # Create config file
    config_parser['common_DL'] = {
        'benchmark_dataset' : args.benchmark_dataset,
        'network_cls' : args.network_cls,
        'epochs' : args.epochs,
        'meta_learning_rate' : args.meta_learning_rate,
    }

    config_parser['MetaLearning'] = {
        'n' : args.n,
        'k' : args.k,
        'meta_batch_size' : args.meta_batch_size,
        'num_steps_ml' : args.num_steps_ml,
        'lr_inner_ml' : args.lr_inner_ml,
        'iterations' : args.iterations,
        'num_steps_validation' : args.num_steps_validation,
    }

    config_parser['LogSave'] = {
        'save_after_epochs' : args.save_after_epochs,
        'report_validation_frequency' : args.report_validation_frequency,
        'log_train_images_after_iteration' : args.log_train_images_after_iteration,
    }


    def _get_config_info(args):
        return f'model-{args.network_cls}_' \
            f'mbs-{args.meta_batch_size}_' \
            f'n-{args.n}_' \
            f'k-{args.k}_' \
            f'stp-{args.num_steps_ml}'
    args_path = os.path.join(maml_path, 'args') 
    step4_args_path = os.path.join(args_path, 'step4_{}.ini'.format(_get_config_info(args)))
    with open(step4_args_path, 'w') as f:
        config_parser.write(f)
    print("Step4 args are saved")


    # Check & Load the existance of *.pkl database file
    base_path_step1 = os.path.join(os.getcwd(),
     'dataset/data/ui_output','maml_{}'.format(args.benchmark_dataset), 'step1')
    os.makedirs(base_path_step1, exist_ok=True)
    save_path_step1 = os.path.join(base_path_step1, '{}.pkl'.format(args.benchmark_dataset))



    # Setup paths
    # 1. Step2's database.pkl path
    base_path_step2 = os.path.join(maml_path, 'step2')
    os.makedirs(base_path_step2, exist_ok=True)
    base_dataset_path_step2 = os.path.join(base_path_step2, args.benchmark_dataset)
    os.makedirs(base_dataset_path_step2, exist_ok=True)
    save_path_step2 = os.path.join(base_dataset_path_step2, '{}.pkl'.format(args.benchmark_dataset))
    save_path_json_step2 = os.path.join(base_dataset_path_step2, 'nwaykshot_{}.json'.format(args.benchmark_dataset))

    # 2. Step3's path : to load the model
    base_path_step3 = os.path.join(maml_path, 'step3')
    base_dataset_path_step3 = os.path.join(base_path_step3, args.benchmark_dataset)

    # 2. Step4's path : to load the adapted model
    base_path_step4 = os.path.join(maml_path, 'step4')
    base_dataset_path_step4 = os.path.join(base_path_step4, args.benchmark_dataset)



    # 3. Step5 base path
    base_path_step = os.path.join(maml_path, 'step5')
    os.makedirs(base_path_step, exist_ok=True)

    base_dataset_path = os.path.join(base_path_step, args.benchmark_dataset)
    os.makedirs(base_dataset_path, exist_ok=True)

    if os.path.isfile(save_path_step2):
        print("Load dataset")
        with open(save_path_step2, 'rb') as f:
            database = pickle.load(f)
    else:
        # 데이터셋 객체를 생성합니다.
        # 타입 : tf.data.Dataset
        if args.benchmark_dataset == "omniglot":
            database = OmniglotDatabase(
                # 200831 changed path, add raw_data folder
                raw_data_address="dataset/raw_data/omniglot".replace('/', os.sep),
                random_seed=47,
                num_train_classes=1200,
                num_val_classes=100)

        elif args.benchmark_dataset == "mini_imagenet":
            database=MiniImagenetDatabase(
                # 200831 changed path, add raw_data folder
                raw_data_address="dataset/raw_data/mini_imagenet".replace('/', os.sep),
                random_seed=-1)

    # Save the database file
    with open(save_path_step2, 'wb') as f:
        pickle.dump(database, f) # e.g. for omniglot, ./dataset/data/ui_output/maml/step1/omniglot.pkl
    # -> To laod this file in the next step

    # 모델 객체를 생성합니다.
    if args.network_cls == "omniglot":
        network_cls=OmniglotModel
    elif args.network_cls == "mini_imagenet":
        network_cls=MiniImagenetModel

    # 학습을 위한 클래스를 생성합니다.
    maml = ModelAgnosticMetaLearning(args, database, network_cls, base_dataset_path)
    # args : 파라미터      type : parser.parse_args
    # database : 데이터셋  type : database
    # network_cls : 모델   type : MetaLearning

    # 입력받은 파라미터를 통해 해당 epochs의 저장된 모델을 불러옵니다.
    # epochs : 몇번 학습한 모델을 불러올 지  type : int
    # None일 시 최종 학습한 모델을 불러옵니다.
    
    # 모델 불러오기 위해 checkpoint의 경로를 step4의 경로로 변경
    maml.checkpoint_dir = os.path.join(base_dataset_path_step3, maml.get_config_info(), 'saved_models')
    # print(maml.checkpoint_dir)
    # epoch_count = maml.load_model(epochs = args.epochs)
    # print("Load the {}th epoch model".format(epoch_count))

    # # meta_test 시 입력받은 파라미터를 통해 fine turning할 횟수 만큼 수행합니다.
    # maml.meta_test(iterations = args.iterations)
    # # iterations : inner loop의 gradient update 횟수 type : int


    # 예측한 결과를 보여줍니다.
    # maml.predict_with_support(save_path=base_dataset_path, meta_test_path=os.path.join('dataset','data','mini_imagenet','test'))
    # meta_test_path 예측할 데이터의 경로               type : string
    # epochs_to_load_from 몇번 학습한 모델을 볼러올지   type : int  None일 시 최종 학습한 모델을 불러옵니다.
    # iterations fine turning 횟수                    type : int
    # print(base_dataset_path)
    # print(save_path_json_step2)

    maml.meta_predict(save_path=base_dataset_path, step2_task_json_path=save_path_json_step2)