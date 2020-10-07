from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # User Input Argument : --benchmark_dataset : omniglot or mini_imagenet
    parser.add_argument('--benchmark_dataset', type=str, default='omniglot')
    args = parser.parse_args()

    # Create Database Object
    # DataType : tf.data.Dataset
    if args.benchmark_dataset == "omniglot":
        database = OmniglotDatabase(
		     # 200831 changed path, add raw_data folder
            raw_data_address="dataset\\raw_data\\omniglot",
            random_seed=47,
            num_train_classes=1200,
            num_val_classes=100,
            is_preview=True)

    # [TODO] Add get_statistic() method to MiniImagenetDatabase class
    elif args.benchmark_dataset == "mini_imagenet":
        database=MiniImagenetDatabase(
		    # 200831 changed path, add raw_data folder
            raw_data_address="dataset\\raw_data\\mini_imagenet",
            random_seed=-1)
    
    database.get_statistic()
    # [TODO] Is it needed to save Database object as pickle file?
    # -> To laod this file in the next step