from model.optimization_based.maml import OmniglotModel, MiniImagenetModel, ModelAgnosticMetaLearning
from model.hetero.mcnn import Hetero,Modified_m_CNN
from dataset.data_generator_oxfordflower import OmniglotDatabase, MiniImagenetDatabase, OxfordFlower
import argparse

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    # 학습에 필요한 파라미터를 입력으로 받습니다.
    parser = argparse.ArgumentParser()
    # parser.add_argument('--benchmark_dataset', type=str, default='mini_imagenet')
    # parser.add_argument('--network_cls', type=str, default='mini_imagenet')
    parser.add_argument('--benchmark_dataset', type=str, default='oxford_flower')
    parser.add_argument('--network_cls', type=str, default='modified_mcnn')
    # parser.add_argument('--n', type=int, default=5) # 필요없음
    parser.add_argument('--epochs', type=int, default=5) # 중복됌
    # parser.add_argument('--iterations', type=int, default=5) # 필요없음
    # parser.add_argument('--k', type=int, default=1) # 필요없음
    # parser.add_argument('--meta_batch_size', type=int, default=32)
    # parser.add_argument('--meta_batch_size', type=int, default=2)
    # parser.add_argument('--num_steps_ml', type=int, default=10)
    parser.add_argument('--lr_inner_ml', type=float, default=0.4)
    # parser.add_argument('--num_steps_validation', type=int, default=10)
    # parser.add_argument('--save_after_epochs', type=int, default=500)
    parser.add_argument('--save_after_epochs', type=int, default=1)
    # parser.add_argument('--meta_learning_rate', type=float, default=0.001)
    parser.add_argument('--report_validation_frequency', type=int, default=50)
    parser.add_argument('--log_train_images_after_iteration', type=int, default=1)
    
    # MultiModal Argument
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--batch_size2",type=int,default=64)# image encoder을 pretrain시킬때의 batch_size
    #parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--epochs2",type=int,default=5) # image encoder을 pretrain시킬때의 epochs

    parser.add_argument("--encoder_name",type=str,default="MobileNetV2")
    parser.add_argument("--optimizer",type=str,default="SGD")

    parser.add_argument("--binary",type=bool,default=False) # is binary classification?
    parser.add_argument("--pretrain",type=bool,default=True) # does pretrain 
    parser.add_argument("--fineTune",type=bool,default=True)
    
    
    args = parser.parse_args()

    # 1 & 2. Preview & Preprocessing
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
    elif args.benchmark_dataset == "oxford_flower":
        database = OxfordFlower(
            config_path="./dataset/data/oxfordflower/args.ini",
            random_seed=47)
            
    # 3. Training 
    # 모델 객체를 생성합니다.
    if args.network_cls == "omniglot":
        network_cls=OmniglotModel
    elif args.network_cls == "mini_imagenet":
        network_cls=MiniImagenetModel
    elif args.network_cls == "modified_mcnn":
        network_cls=Modified_m_CNN
        
    if network_cls in [OmniglotModel,MiniImagenetModel]:
        maml = ModelAgnosticMetaLearning(args, database, network_cls)
        maml.meta_train(epochs = args.epochs)
    elif network_cls in [Modified_m_CNN]:
        hetero = Hetero(args,"./dataset/data/oxfordflower/args.ini",database,network_cls)
        hetero.train()

    # 4. Test (Evaluation)
    # Model Load : [TODO] Load from ckpt is required...
    # 각 단계별로 코드 파일을 분할해야하기 때문에 저장된 모델을 불러오는 기능 필요
    # input으로 epochs를 받아 해당 epoch에 저장된 모델 불러오기
    # None일 시 최종 학습한 모델을 불러옵니다.
    # hetero.load_model()

    # Evaluation
    hetero.evaluate() # [TODO] Change the method name -> hetero.test()

    # 5. Prediction
    # [TODO] input으로 path를 받도록 변경 (현재는 dictionary)
    # hetero.predict()


"""
    # 학습을 위한 클래스를 생성합니다.
    maml = ModelAgnosticMetaLearning(args, database, network_cls)
    # args : 파라미터      type : parser.parse_args
    # database : 데이터셋  type : database
    # network_cls : 모델   type : MetaLearning
    
    print("=======================meta TRAIN")
    # 학습을 위한 클래스를 사용하여 입력받은 파라미터를 통해 meta_train을 수행합니다.
    maml.meta_train(epochs = args.epochs)
    # epochs : 반복 횟수 type : int

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
"""
