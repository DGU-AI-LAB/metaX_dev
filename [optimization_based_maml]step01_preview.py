from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse
import logging, os
import pickle

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # User Input Argument : --benchmark_dataset : omniglot or mini_imagenet
    parser.add_argument('--benchmark_dataset', type=str, default='omniglot')
    args = parser.parse_args()

    # Check & Load the existance of *.pkl database file
    base_path = os.path.join(os.getcwd(),
     'dataset/data/ui_output','maml_{}'.format(args.benchmark_dataset), 'step1')
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, '{}.pkl'.format(args.benchmark_dataset))
    if os.path.isfile(save_path):
        print("Load dataset")
        with open(save_path, 'rb') as f:
            database = pickle.load(f)

    else:
        # Create Database Object
        # DataType : tf.data.Dataset
        if args.benchmark_dataset == "omniglot":
            database = OmniglotDatabase(
                # 200831 changed path, add raw_data folder
                raw_data_address="dataset/raw_data/omniglot",
                random_seed=47,
                num_train_classes=1200,
                num_val_classes=100,
                is_preview=False)

        # [TODO] Add get_statistic() method to MiniImagenetDatabase class
        elif args.benchmark_dataset == "mini_imagenet":
            database=MiniImagenetDatabase(
                # 200831 changed path, add raw_data folder
                raw_data_address="dataset/raw_data/mini_imagenet",
                random_seed=-1)

        # Save the database file
        with open(save_path, 'wb') as f:
            pickle.dump(database, f) # e.g. for omniglot, ./dataset/data/ui_output/maml/step1/omniglot.pkl
        # -> To laod this file in the next step
    
    # This code saves the stat of the dataset and the file path of each class
    database.is_preview = True
    database.get_statistic(base_path=base_path)