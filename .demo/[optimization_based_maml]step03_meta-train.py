from model.optimization_based.maml import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearning
from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse

import logging, os, glob
import pickle
from configparser import ConfigParser

config = ConfigParser()

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    # User Input of STEP3
    # 1. epochs : 학습시 전체 데이터셋 반복 횟수
    # 2. network_cls : 메타러닝으로 학습될 모델 기본값 : step2에서 저장된 값
    # 3. meta_learning_rate : Outer loop learning rate

    # 4. meta_batch_size : N of tasks in one meta-batch (2단계에서 저장된 값을 무시하고 재설정가능)
    # 5. num_steps_ml : Training set에 대한 inner loop gradient update step 횟수
    # 6. lr_inner_ml : Inner loop의 learning rate
    # 7. num_steps_validation : Validation set에 대한  inner loop gradient update step 횟수

    # 8. save_after_epochs : 모델 저장 주기(1인 경우 매 epoch마다 저장)
    # 9. report_validation_frequency : validation set에 대한 evaluation 결과 프린트 주기

    parser = argparse.ArgumentParser()

    # Load latest step2_{config}.ini
    config_parser = ConfigParser()
    maml_path = os.path.join(os.getcwd(), 'dataset/data/ui_output'.replace('/', os.sep),'maml')
    
    # 학습이 이미 완료된 모델을 강제로 불러오는 코드
    manually_load_file = None
    # 아래를 이용해 불러오고자하는 모델의 setp3_[모델세팅].ini 파일을 불러옵니다.
    # manually_load_file = os.path.join(maml_path, 'args', 'step3_model-omniglot_mbs-2_n-5_k-3_stp-1.ini')
    
    if not manually_load_file:
        # 이전 스텝의 args를 기본값으로 갖도록 등록
        args_path = os.path.join(maml_path, 'args', '*') 
        list_of_args_ini_files = glob.glob(args_path)
        list_of_args_ini_files = [i for i in list_of_args_ini_files if 'step3_' in i]
        latest_file = max(list_of_args_ini_files, key=os.path.getctime)
        print("Load Step2 arguments from : {}".format(latest_file))
        config_parser.read(latest_file)
    else:
        print("Load Step3 arguments from : {}".format(manually_load_file))
        config_parser.read(manually_load_file)
    

    # 빠른 테스트를 위한 세팅
    # Argument for Common Deep Learning
    # It take user input & It also have default values
    parser.add_argument('--benchmark_dataset', type=str, default=config_parser['common_DL']['benchmark_dataset'])
    parser.add_argument('--network_cls', type=str, default=config_parser['common_DL']['benchmark_dataset']) # User Input STEP3
    parser.add_argument('--epochs', type=int, default=20)          # User Input STEP3
    parser.add_argument('--meta_learning_rate', type=float, default=0.001) # User Input STEP3

    # Argument for Meta-Learning
    # It take user input & It also have default values
    parser.add_argument('--n', type=int, default=config_parser['MetaLearning']['n']) 
    parser.add_argument('--k', type=int, default=config_parser['MetaLearning']['k'])
    parser.add_argument('--meta_batch_size', type=int, default=2)  # User Input STEP3
    parser.add_argument('--num_steps_ml', type=int, default=1)     # User Input STEP3
    parser.add_argument('--lr_inner_ml', type=float, default=0.4)  # User Input STEP3
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--num_steps_validation', type=int, default=1) # User Input STEP3
    # --iterations : STEP3 에서 사용되진 않지만 다음 스텝에서 모델로드시 변수가 존재해야 하므로 임의값 할당

    # Argument for wrting log & save file
    # It take user input & It also have default values
    parser.add_argument('--save_after_epochs', type=int, default=1) # User Input STEP3
    parser.add_argument('--report_validation_frequency', type=int, default=1) # User Input STEP3
    parser.add_argument('--log_train_images_after_iteration', type=int, default=-1) # No Record


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
        # 'iterations' : args.iterations,
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

    step3_args_path = os.path.join(args_path, 'step3_{}.ini'.format(_get_config_info(args)))
    with open(step3_args_path, 'w') as f:
        config_parser.write(f)
    print("Step3 args are saved")


    # Setup paths
    # 1. Step2's database.pkl path
    base_path_step2 = os.path.join(maml_path, 'step2')
    os.makedirs(base_path_step2, exist_ok=True)
    base_dataset_path_step2 = os.path.join(base_path_step2, args.benchmark_dataset)
    os.makedirs(base_dataset_path_step2, exist_ok=True)
    save_path_step2 = os.path.join(base_dataset_path_step2, '{}.pkl'.format(args.benchmark_dataset))

    # 2. Step3 base path
    base_path_step = os.path.join(maml_path, 'step3')
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
                raw_data_address="dataset/aw_data/mini_imagenet".replace('/', os.sep),
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
    
    # 학습을 위한 클래스를 사용하여 입력받은 파라미터를 통해 meta_train을 수행합니다.
    maml.meta_train(epochs = args.epochs)
    # epochs : 반복 횟수 type : int