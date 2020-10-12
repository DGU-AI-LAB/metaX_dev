from model.optimization_based.maml import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearning
from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse

import logging, os

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
    parser.add_argument('--epochs', type=int, default=3) # 5
    parser.add_argument('--iterations', type=int, default=1) # 5
    parser.add_argument('--k', type=int, default=1)
    # parser.add_argument('--meta_batch_size', type=int, default=32)
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
    # # parser.add_argument('--benchmark_dataset', type=str, default='mini_imagenet')
    # # parser.add_argument('--network_cls', type=str, default='mini_imagenet')

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
            raw_data_address="dataset/raw_data/mini_imagenet",
            random_seed=-1)

    # -> N-way K-shot 프리뷰 보여줄 수 있게 코드 추가

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
    
    # 학습을 위한 클래스를 사용하여 입력받은 파라미터를 통해 meta_train을 수행합니다.
    maml.meta_train(epochs = args.epochs)
    # epochs : 반복 횟수 type : int

    # meta_test 시 입력받은 파라미터를 통해 fine turning할 횟수 만큼 수행합니다.
    maml.meta_test(iterations = args.iterations)
    # iterations : inner loop의 gradient update 횟수 type : int

    # 입력받은 파라미터를 통해 해당 epochs의 저장된 모델을 불러옵니다.
    maml.load_model(epochs = args.epochs)
    print(maml)
    # epochs : 몇번 학습한 모델을 불러올 지  type : int
    # None일 시 최종 학습한 모델을 불러옵니다.

    # 예측한 결과를 보여줍니다.
    # print(maml.predict_with_support(meta_test_path='/dataset/data/omniglot_test'))
    # meta_test_path 예측할 데이터의 경로               type : string
    # epochs_to_load_from 몇번 학습한 모델을 볼러올지   type : int  None일 시 최종 학습한 모델을 불러옵니다.
    # iterations fine turning 횟수                    type : int