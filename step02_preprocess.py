from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    # 빠른 테스트를 위한 세팅
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dataset', type=str, default='omniglot')       # 20.09.03
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=2)  # 20.09.03


    args = parser.parse_args()

    # 데이터셋 객체를 생성합니다.
    # 타입 : tf.data.Dataset
    if args.benchmark_dataset == "omniglot":
        database = OmniglotDatabase(
		     # 200831 changed path, add raw_data folder
            raw_data_address="dataset\raw_data\omniglot",
            random_seed=47,
            num_train_classes=1200,
            num_val_classes=100)
    elif args.benchmark_dataset == "mini_imagenet":
        database=MiniImagenetDatabase(
		    # 200831 changed path, add raw_data folder
            raw_data_address="\dataset\raw_data\mini_imagenet",
            random_seed=-1)

    # -> N-way K-shot 프리뷰 보여줄 수 있게 코드 추가