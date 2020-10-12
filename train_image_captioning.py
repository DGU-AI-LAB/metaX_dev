from metaX.models.optimization_based.MAML import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearning
from metaX.dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase

from metaX.models.heterogeneous_data_analysis.ImageCaptioningModel import ImageCaptioningModel
from metaX.dataset.MSCOCOKR_data_generator import MSCOCOKRDatabase
import argparse

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    # 학습에 필요한 파라미터를 입력으로 받습니다.
    parser = argparse.ArgumentParser()
    # parser.add_argument('--benchmark_dataset', type=str, default='mini_imagenet')
    # parser.add_argument('--network_cls', type=str, default='mini_imagenet')
    parser.add_argument('--benchmark_dataset', type=str, default='omniglot')
    parser.add_argument('--network_cls', type=str, default='omniglot')
    # parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=5)
    # parser.add_argument('--k', type=int, default=1)
    # parser.add_argument('--meta_batch_size', type=int, default=32)
    # parser.add_argument('--meta_batch_size', type=int, default=2)
    parser.add_argument('--num_steps_ml', type=int, default=10)
    parser.add_argument('--lr_inner_ml', type=float, default=0.4)
    parser.add_argument('--num_steps_validation', type=int, default=10)
    # parser.add_argument('--save_after_epochs', type=int, default=500)
    parser.add_argument('--save_after_epochs', type=int, default=1)
    parser.add_argument('--meta_learning_rate', type=float, default=0.001)
    parser.add_argument('--report_validation_frequency', type=int, default=50)
    parser.add_argument('--log_train_images_after_iteration', type=int, default=1)
    
    # ImageCaptioning 
    parser.add_argument('--Embedding_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.001)


    args = parser.parse_args()

    # 데이터셋 객체를 생성합니다.
    # 타입 : tf.data.Dataset
    if args.benchmark_dataset == "omniglot":
        database = OmniglotDatabase(
            raw_data_address="dataset\omniglot",
            random_seed=47,
            num_train_classes=1200,
            num_val_classes=100)
    elif args.benchmark_dataset == "mini_imagenet":
        database=MiniImagenetDatabase(
            raw_data_address="\dataset\mini_imagenet",
            random_seed=-1)
    elif args.benchmark_dataset == "MSKOKOKR":
        database = MSCOCOKRDatabase(
            train_address = "dataset\MSCOCOKR_data\train",
            test_address = "dataset\MSCOCOKR_data\test")

    # 모델 객체를 생성합니다.
    if args.network_cls == "omniglot":
        network_cls=OmniglotModel
    elif args.network_cls == "mini_imagenet":
        network_cls=MiniImagenetModel
    elif args.network_cls == "mscoco_kor":
        network_cls = ImageCaptioningModel

    # 학습을 위한 클래스를 생성합니다.
    #maml = ModelAgnosticMetaLearning(args, database, network_cls)
    # args : 파라미터      type : parser.parse_args
    # database : 데이터셋  type : database
    # network_cls : 모델   type : MetaLearning
    
    imgcap = ImageCaptioningModel(args, database, network_cls)
    
    
    # print("=======================meta TRAIN")
    # 학습을 위한 클래스를 사용하여 입력받은 파라미터를 통해 meta_train을 수행합니다.
    # maml.meta_train(epochs = args.epochs)
    # epochs : 반복 횟수 type : int
    
    print("=========================TRAIN")
    imgcap.train(epochs = args.epochs, iterations = args.iterations)
    
    imgcap.evaluate()
    
    imgcap.predict()
    
    
    # # meta_test 시 입력받은 파라미터를 통해 fine turning할 횟수 만큼 수행합니다.
    # maml.meta_test(iterations = args.iterations)
    # # iterations : inner loop의 gradient update 횟수 type : int

    # # 입력받은 파라미터를 통해 해당 epochs의 저장된 모델을 불러옵니다.
    # maml.load_model(epochs = args.epochs)
    # # epochs : 몇번 학습한 모델을 불러올 지  type : int
    # # None일 시 최종 학습한 모델을 불러옵니다.

    # # 예측한 결과를 보여줍니다.
    # print(maml.predict_with_support(meta_test_path='/dataset/data/mini_imagenet/test'))
    # # meta_test_path 예측할 데이터의 경로               type : string
    # # epochs_to_load_from 몇번 학습한 모델을 볼러올지   type : int  None일 시 최종 학습한 모델을 불러옵니다.
    # # iterations fine turning 횟수                    type : int