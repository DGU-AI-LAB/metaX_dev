from dataset.data_generator import OmniglotDatabase, MiniImagenetDatabase
import argparse
import logging, os
import pickle
import tqdm
import json
import numpy as np
from configparser import ConfigParser

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def save_nwaykshot(dataset, save_path, class2num):
    dataset = list(dataset)
    dataset= np.array(dataset)
    # dataset.shape
    # (N of meta-batch, data&label, meta-batch, support&query, N, K)
    # - meta-batch : Number of N-way K-shot Tasks per outer loop update
    # - number of tasks : N of meta-batch * meta-batch

    # Select only data part
    # (N of meta-batch, meta-batch, support&query, N, K)
    dataset = dataset[:,0,:,:,:, :]
    # Flatten the meta-batches
    # (N of meta-batch * meta-batch, support&query, N, K)
    dataset = dataset.reshape(-1, *dataset.shape[2:])

    json_file = {}
    logging.info("Preparing N-way K-shot json files")
    for i, task in tqdm.tqdm(enumerate(dataset)):
        key_name = 'task{}'.format(i+1)
        # Task
        json_file[key_name] = {'supports' : {},
                            "query"    : {} }
        
        support, query = task
        
        for n_class in support:
            class_path = bytes.decode(n_class[0].numpy())
            classnum = class2num[class_path.split(os.sep)[-2]]
            json_file[key_name]['supports'][classnum] = []

            for k_sample in n_class:
                path = bytes.decode(k_sample.numpy())
                name = path.split(os.sep)[-1]

                json_file[key_name]['supports'][classnum].append(
                    {'name' : name, 'path' : path}
                )

        for n_class in query:
            # print("n_class")
            class_path = bytes.decode(n_class[0].numpy())
            classnum = class2num[class_path.split(os.sep)[-2]]
            json_file[key_name]['query'][classnum] = []

            for k_sample in n_class:
                path = bytes.decode(k_sample.numpy())
                name = path.split(os.sep)[-1]

                json_file[key_name]['query'][classnum].append(
                    {'name' : name, 'path' : path}
                )

    logging.info("Start Saving N-way K-shot json file")
    print("the N-way K-Shot JSON file has been saved in ", save_path)
    with open(save_path, 'w') as f:
        json.dump(json_file, f, indent='\t')
    logging.info("Saving completed.")



if __name__ == '__main__':
    # User Input Argument
    # - n : The number of the sampled class
    # - k : The number of the samples per each class
    # - meta_batch_size : The number of the n-way k-shot tasks in one batch
    parser = argparse.ArgumentParser()
    
    # Config File Load : Step 1 Config file
    config_parser = ConfigParser()
    maml_path = os.path.join(os.getcwd(), 'dataset/data/ui_output','maml')
    args_path = os.path.join(maml_path, 'args') 
    step1_args_path = os.path.join(args_path, 'step1.ini')
    config_parser.read(step1_args_path)
    print("Load Step1 arguments from : {}".format(step1_args_path))

    # Config File Writing and save : Step 2 Config file
    parser.add_argument('--benchmark_dataset', type=str, default=config_parser['common_DL']['benchmark_dataset'])       # 20.09.03
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=2)  # 20.09.03
    args = parser.parse_args()

    config_parser['MetaLearning'] = {
        'n' : args.n,
        'k' : args.k,
        'meta_batch_size' : args.meta_batch_size
        }
    
    step2_args_path = os.path.join(args_path, 'step2.ini')
    with open(step2_args_path, 'w') as f:
        config_parser.write(f)
    print("Step2 args are saved")

    # Setup paths
    # 1. Step1's database.pkl path
    base_path_step1 = os.path.join(maml_path, 'step1')
    os.makedirs(base_path_step1, exist_ok=True)

    base_dataset_path_step1 = os.path.join(base_path_step1, args.benchmark_dataset)
    os.makedirs(base_dataset_path_step1, exist_ok=True)
    save_path_step1 = os.path.join(base_dataset_path_step1, '{}.pkl'.format(args.benchmark_dataset))
    
    # 2. Step2 base path
    base_path_step = os.path.join(maml_path, 'step2')
    os.makedirs(base_path_step, exist_ok=True)

    base_dataset_path = os.path.join(base_path_step, args.benchmark_dataset)
    os.makedirs(base_dataset_path, exist_ok=True)

    save_path = os.path.join(base_dataset_path, '{}.pkl'.format(args.benchmark_dataset))

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
                raw_data_address="/dataset/raw_data/mini_imagenet",
                random_seed=-1)

    # Save the database file
    with open(save_path, 'wb') as f:
        pickle.dump(database, f) # e.g. for omniglot, ./dataset/data/ui_output/maml/step2/omniglot.pkl
    # -> To laod this file in the next step


    # Saving N-way K-shot JSON

    database.is_preview = True

    train_dataset = database.get_supervised_meta_learning_dataset(
        database.train_folders,
        n = args.n,
        k = args.k,
        meta_batch_size = args.meta_batch_size
    )

    ######### This Part Maybe be differnet for miniImageNet dataset #############
    # Numbering the classees
    train_folders = sorted(database.train_folders)
    val_folders = sorted(database.val_folders)
    test_folders = sorted(database.test_folders)

    folders = train_folders + val_folders + test_folders
    folders.sort()
    
    class2num = { i.split(os.sep)[-1]: 'class{}'.format(n) for n, i in enumerate(folders) }
    num2class = {v : k for k, v in class2num.items()}
    ###############################################################################
    
    # Save the N-way K-shot task json file (for tarin set)
    json_save_path = os.path.join(base_dataset_path, 'nwaykshot_{}.json'.format(args.benchmark_dataset))
    save_nwaykshot(train_dataset, json_save_path, class2num)

    # [TODO] For Mini ImageNet Setting