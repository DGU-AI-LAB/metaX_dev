import os
import shutil
from abc import ABC, abstractmethod
import random
import pickle
import utils
import tensorflow as tf
from collections import defaultdict
from glob import glob
from PIL import Image

# Utilities functions for OxfordFlower
from dataset.oxfordflower.data_utils import parse_config, tokenizer,build_vocab,txt2Token,img2Raw,load_json,save_json,match_and_write
from dataset.oxfordflower.tf_utils import config_wrapper_parse_funcs

class Database(ABC):
    def __init__(self, raw_database_address, database_address, random_seed=-1):
        if random_seed != -1:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.raw_database_address = raw_database_address
        self.database_address = database_address

        self.prepare_database()
        self.train_folders, self.val_folders, self.test_folders = self.get_train_val_test_folders()

        self.input_shape = self.get_input_shape()

    @abstractmethod
    def get_class(self):
        pass

    @abstractmethod
    def preview_image(self):
        pass

    @abstractmethod
    def get_input_shape(self):
        pass

    @abstractmethod
    def prepare_database(self):
        pass

    @abstractmethod
    def get_train_val_test_folders(self):
        pass

    def check_number_of_samples_at_each_class_meet_minimum(self, folders, minimum):
        for folder in folders:
            if len(os.listdir(folder)) < 2 * minimum:
                raise Exception(
                    f'There should be at least {2 * minimum} examples in each class. Class {folder} does not have that many examples')

    def _get_instances(self, k):
        def get_instances(class_dir_address):
            return tf.data.Dataset.list_files(class_dir_address, shuffle=True).take(2 * k)

        return get_instances

    def _get_parse_function(self):
        def parse_function(example_address):
            return example_address

        return parse_function

    def make_labels_dataset(self, n, k, meta_batch_size, steps_per_epoch, one_hot_labels):
        labels_dataset = tf.data.Dataset.range(n)
        if one_hot_labels:
            labels_dataset = labels_dataset.map(lambda example: tf.one_hot(example, depth=n))

        labels_dataset = labels_dataset.interleave(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(2 * k),
            cycle_length=n,
            block_length=k
        )
        labels_dataset = labels_dataset.repeat(meta_batch_size)
        labels_dataset = labels_dataset.repeat(steps_per_epoch)
        return labels_dataset

    def get_supervised_meta_learning_dataset(
            self,
            folders,
            n,
            k,
            meta_batch_size,
            one_hot_labels=True,
            reshuffle_each_iteration=True,
    ):
        for class_name in folders:
            assert (len(os.listdir(class_name)) > 2 * k), f'The number of instances in each class should be larger ' \
                f'than {2 * k}, however, the number of instances in' \
                f' {class_name} are: {len(os.listdir(class_name))}'

        classes = [class_name + '/*' for class_name in folders]
        steps_per_epoch = len(classes) // n // meta_batch_size

        labels_dataset = self.make_labels_dataset(n, k, meta_batch_size, steps_per_epoch, one_hot_labels)

        dataset = tf.data.Dataset.from_tensor_slices(classes)
        # print(len(folders))
        dataset = dataset.shuffle(buffer_size=len(folders), reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.interleave(
            self._get_instances(k),
            cycle_length=n,
            block_length=k,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(self._get_parse_function(), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = tf.data.Dataset.zip((dataset, labels_dataset))

        dataset = dataset.batch(k, drop_remainder=False)
        dataset = dataset.batch(n, drop_remainder=True)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.batch(meta_batch_size, drop_remainder=True)

        setattr(dataset, 'steps_per_epoch', steps_per_epoch)
        return dataset

class OmniglotDatabase(Database):
    def __init__(
            self,
            raw_data_address,
            random_seed,
            num_train_classes,
            num_val_classes,
    ):
        self.num_train_classes = num_train_classes
        self.num_val_classes = num_val_classes

        super(OmniglotDatabase, self).__init__(
            raw_data_address,
            os.getcwd()+'/dataset/data/omniglot',
            random_seed=random_seed,
        )

    def get_class(self):
        train_dict = defaultdict(list)
        val_dict = defaultdict(list)
        test_dict = defaultdict(list)
        for train_class in self.train_folders:
            for train_image_path in glob(train_class + '/*.*'):
                train_dict[train_class.split('\\')[-1]].append(train_image_path)

        for val_class in self.train_folders:
            for val_image_path in glob(val_class+'/*.*'):
                val_dict[val_class.split('\\')[-1]].append(val_image_path)

        for test_class in self.train_folders:
            for test_image_path in glob(test_class+'/*.*'):
                test_dict[test_class.split('\\')[-1]].append(test_image_path)

        return train_dict, val_dict, test_dict

    def preview_image(self, image_path):
        image = Image.open(image_path)
        # image.show()

        return image

    def get_input_shape(self):
        return 28, 28, 1

    def get_train_val_test_folders(self):
        num_train_classes = self.num_train_classes
        num_val_classes = self.num_val_classes

        folders = [os.path.join(self.database_address, class_name) for class_name in os.listdir(self.database_address)]
        folders.sort()
        random.shuffle(folders)
        train_folders = folders[:num_train_classes]
        val_folders = folders[num_train_classes:num_train_classes + num_val_classes]
        test_folders = folders[num_train_classes + num_val_classes:]

        # print(len(train_folders))

        return train_folders, val_folders, test_folders

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (28, 28))
            image = tf.cast(image, tf.float32)

            return 1 - (image / 255.)

        return parse_function

    def prepare_database(self):
        for item in ('images_background', 'images_evaluation'):
            alphabets = os.listdir(os.path.join(self.raw_database_address, item))
            for alphabet in alphabets:
                alphabet_address = os.path.join(self.raw_database_address, item, alphabet)
                for character in os.listdir(alphabet_address):
                    character_address = os.path.join(alphabet_address, character)
                    destination_address = os.path.join(self.database_address, alphabet + '_' + character)
                    if not os.path.exists(destination_address):
                        shutil.copytree(character_address, destination_address)


class MiniImagenetDatabase(Database):
    def __init__(self, raw_data_address, random_seed=-1, config=None):
        super(MiniImagenetDatabase, self).__init__(
            raw_data_address,
            os.getcwd() + '/dataset/data/mini_imagenet',
            random_seed=random_seed,
        )

    def get_class(self):
        train_dict = defaultdict(list)
        val_dict = defaultdict(list)
        test_dict = defaultdict(list)
        for train_class in self.train_folders:
            for train_image_path in glob(train_class + '/*.*'):
                train_dict[train_class.split('\\')[-1]].append(train_image_path)

        for val_class in self.train_folders:
            for val_image_path in glob(val_class + '/*.*'):
                val_dict[val_class.split('\\')[-1]].append(val_image_path)

        for test_class in self.train_folders:
            for test_image_path in glob(test_class + '/*.*'):
                test_dict[test_class.split('\\')[-1]].append(test_image_path)

        return train_dict, val_dict, test_dict

    def preview_image(self, image_path):
        image = Image.open(image_path)
        image.show()

        return image

    def get_input_shape(self):
        return 84, 84, 3

    def get_train_val_test_folders(self):
        dataset_folders = list()
        for dataset_type in ('train', 'val', 'test'):
            dataset_base_address = os.path.join(self.database_address, dataset_type)
            folders = [
                os.path.join(dataset_base_address, class_name) for class_name in os.listdir(dataset_base_address)
            ]
            dataset_folders.append(folders)
        return dataset_folders[0], dataset_folders[1], dataset_folders[2]

    def _get_parse_function(self):
        def parse_function(example_address):
            image = tf.image.decode_jpeg(tf.io.read_file(example_address))
            image = tf.image.resize(image, (84, 84))
            image = tf.cast(image, tf.float32)

            return image / 255.

        return parse_function

    def prepare_database(self):
        if not os.path.exists(self.database_address):
            shutil.copytree(self.raw_database_address, self.database_address)



class OxfordFlower(Database):
    """
    Database for OxfordFlower classification.
    Preprocess raw dataset into tfrecords, in which (image,text) pairs are saved as bytes string.
    If tfrecords already exist, __init__ (getting an instace) does nothing.
    
    NOTE) Utility functions for processing text,image are in 'dataset.oxfordflower.data_utils' and 'dataset.oxfordflower.tf_utils'
          
    """   
    def __init__(self,config_path=None,random_seed=1234,ratio=(0.8,0.1,0.1)):
        """
        Split whole data into train, eval and test dataset.
        Construct TFRecords from raw data (Images and Texts).
        """
        self.config = parse_config(config_path)
        self.config["seed"]= random_seed
        
        self.raw_database_address = self.config["base_path"]
        self.database_address = self.config["data_path"]
        
        super(OxfordFlower,self).__init__(
            self.raw_database_address,
            self.database_address,
            random_seed
        )
                        
    def prepare_database(self):
        """
        Save Train, Evaluation, Test datasets as tfrecord files.
        """
        config = self.config        
        # Create directories if not exists
        #if not os.path.exists(os.path.join(config["data_path"],config["tfrecord_path"])):
        os.makedirs(os.path.join(config["data_path"],config["tfrecord_path"]),exist_ok=True)
        #if not os.path.exists(os.path.join(config["data_path"],config["pretrain_path"])):
        os.makedirs(os.path.join(config["data_path"],config["pretrain_path"]),exist_ok=True)
        
        if glob(os.path.join(config["data_path"],"tfrecord","*.record")):
            print("Use Existing TFRecord files")
            return True
        else:
            pass
        
        # Load train,eval,test dictionary (Names does not make sense, but for compatibility)
        self.train_folders, self.val_folders, self.test_folders = \
            self.get_train_val_test_folders2(ratio=(0.8,0.1,0.1))
        
        #train_dict = load_json(os.path.join(config["base_path"],config["train_json"]))
        #eval_dict = load_json(os.path.join(config["base_path"],config["eval_json"]))
        #test_dict = load_json(os.path.join(config["base_path"],config["test_json"]))
        _tokenizer = tokenizer("Okt") # argparser로 줄것 
        
        # build and save vocab
        vocab=build_vocab(config,_tokenizer)
        with open(os.path.join(config["data_path"],config["vocab"]),"wb") as f:
            pickle.dump(vocab,f)
        
        # Tokenize according to a vocab
        name2token = txt2Token(config,_tokenizer,vocab)
        
        # Encode images to raw byte string
        name2img = img2Raw(config)
        
        # Creator TFRecordWriter
        train_tfwriter=tf.io.TFRecordWriter(
            os.path.join(config["data_path"],config["tfrecord_path"],"train.record"))
        eval_tfwriter=tf.io.TFRecordWriter(
           os.path.join(config["data_path"],config["tfrecord_path"],"eval.record"))
        test_tfwriter=tf.io.TFRecordWriter(
            os.path.join(config["data_path"],config["tfrecord_path"],"test.record"))    
        
        # Write according to train,eval,test dictionary
        for _id, example in self.train_folders.items():
            match_and_write(_id,example,name2img,name2token,train_tfwriter)
        train_tfwriter.close()
            
        for _id, example in self.val_folders.items():
            match_and_write(_id,example,name2img,name2token,eval_tfwriter)
        eval_tfwriter.close()
            
        for _id, example in self.test_folders.items():
            match_and_write(_id,example,name2img,name2token,test_tfwriter)
        test_tfwriter.close()
        
    def get_train_val_test_folders(self):
        
        trn_dict,eval_dict,test_dict =self.return_dict()

        if any((trn_dict,eval_dict,test_dict)) is None:
            raise ValueError("One of Attributes not exists : 'train_folders','eval_folders','test_folders'")
        else:
            return trn_dict,eval_dict,test_dict      
        
    def return_dict(self):
        """If any, return train,eval,test dataset dictionaies"""
        config = self.config
        if os.path.exists(os.path.join(config["base_path"],config["train_json"])):
            if os.path.exists(os.path.join(config["base_path"],config["eval_json"])):
                if os.path.exists(os.path.join(config["base_path"],config["test_json"])):

                    train_dict = load_json(os.path.join(config["base_path"],config["train_json"]))
                    eval_dict = load_json(os.path.join(config["base_path"],config["eval_json"]))
                    test_dict = load_json(os.path.join(config["base_path"],config["test_json"]))               
                    
                    return train_dict , eval_dict , test_dict
        
        print("At least, one of train,eval,test dictionaries do not exist")
        return None,None,None
    
    def get_train_val_test_folders2(self,seed=1234,ratio=(0.8,0.1,0.1)):
        """
        For multi-view training, Split the whole dataset into train,evaluation and test, and return path of training,evaluation,test json file
        """
        
        config = self.config
        
        trn_dict,eval_dict,test_dict =self.return_dict()
        if any((trn_dict,eval_dict,test_dict)) is None:
            pass
        else:
            return trn_dict,eval_dict,test_dict  
         
        img_path = os.path.join(self.raw_database_address,"images")
        txt_path = os.path.join(self.raw_database_address,"texts")
        
        # Check if data valid
        assert len(os.listdir(img_path)) == len(os.listdir(txt_path)),"Num classes differs"
        
        total_num = 0
        for fdir in os.listdir(img_path):
            total_num+=len(os.listdir(os.path.join(img_path,fdir)))

        print("Total Num :",total_num)
        
        whole_imgs = []
        for fdir in os.listdir(img_path):
            imgs = os.listdir(os.path.join(img_path,fdir))
            for img in imgs:
                whole_imgs.append((int(fdir),img))    
        whole_imgs = sorted(whole_imgs,key=lambda x:x[-1])

        whole_txts = []
        for fdir in os.listdir(txt_path):
            txts = os.listdir(os.path.join(txt_path,fdir))
            for txt in txts:
                whole_txts.append((int(fdir),txt))    
        whole_txts = sorted(whole_txts,key=lambda x:x[-1])

        whole_lst = []
        for (cls,img),(cls2,txt)in zip(whole_imgs,whole_txts):
            assert cls == cls2
            whole_lst.append((cls,img,txt))
        
        random.shuffle(whole_lst)
        
        trn_num  = int(total_num*ratio[0])
        eval_num = int(total_num*ratio[1])
        test_num = total_num - trn_num - eval_num
        assert trn_num+eval_num+test_num == total_num

        trn_dict = {}
        eval_dict = {}
        test_dict = {}

        for i,(cls,img,txt) in enumerate(whole_lst):
    
            if i < trn_num:
                trn_dict[str(i)] = {"class":cls,"img_file":img,"txt_file":txt}
            elif i < trn_num+eval_num:
                eval_dict[str(i)] = {"class":cls,"img_file":img,"txt_file":txt}
            else:
                test_dict[str(i)] = {"class":cls,"img_file":img,"txt_file":txt}

        save_json(trn_dict,os.path.join(config["base_path"],config["train_json"]))
        save_json(eval_dict,os.path.join(config["base_path"],config["eval_json"]))
        save_json(test_dict,os.path.join(config["base_path"],config["test_json"]))

        return trn_dict,eval_dict,test_dict
    
    def get_input_shape(self):
        return dict(
            Image_size = self.config["img_size"],
            Text_size= self.config["max_len"])
    
    def preview_image(self,class_num=None,id_num=None):
        
        if class_num is None:
            classes = list(range(0,102))
            random.shuffle(classes)
            selected_class = classes[0]
        else:
            selected_class = class_num
            
        text_dir = os.path.join(self.config["base_path"],"texts",str(selected_class))
        img_dir = os.path.join(self.config["base_path"],"images",str(selected_class))
            
        text_files = sorted(glob(text_dir+"/*"))
        image_files = sorted(glob(img_dir+"/*"))
            
        assert len(text_files) == len(image_files)
        
        if id_num is None:
            selected_case = list(range(0,len(text_files)))
            random.shuffle(selected_case)
            selected_id = selected_case[0]
        else:
            selected_id = id_num
            
        img = image_files[selected_id]
        txt = text_files[selected_id]
            
        img = Image.open(img)
        with open(txt,"r") as f:
            txt = f.read()
            
        return {"class":selected_class,"image":img,"text":txt}
        
    def get_class(self):
        flower_names = [
         'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
         'tiger lily' , 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot",
         'king protea', 'spear thistle','yellow iris','globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
         'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy',
         'prince of wales feathers','stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox',
         'love in the mist','mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
         'siam tulip','lenten rose','barbeton daisy', 'daffodil', 'sword lily','poinsettia', 'bolero deep blue', 'wallflower',
         'marigold','buttercup','oxeye daisy','common dandelion','petunia','wild pansy','primula','sunflower', 'pelargonium',
         'bishop of llandaff','gaura','geranium','orange dahlia','pink-yellow dahlia?','cautleya spicata', 'japanese anemone',
         'black-eyed susan','silverbush','californian poppy','osteospermum','spring crocus','bearded iris','windflower',
         'tree poppy','gazania','azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus',
         'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow',
         'magnolia','cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove',
         'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
          'blackberry lily']
        return {i:name for i,name in zip(range(102),flower_names)}
    
    def data_loader(self,usage,mode,batch_size,shuffle):
        config = self.config
        
        _parse_img_example,_parse_txt_example,_parse_single_example = config_wrapper_parse_funcs(config)
        assert usage in ["train","eval","test"], "mode should be one of 'train','eval','test'"
        
        if mode=="text" or mode=="txt":
            parse_func = _parse_txt_example
        elif mode=="image" or mode=="img":
            parse_func = _parse_img_example
        elif mode=="both" or mode==None:
            parse_func = _parse_single_example
        else:
            parse_func = _parse_single_example
            
        path = os.path.join(config["data_path"],config["tfrecord_path"],f"{usage}.record")
        dataset = tf.data.TFRecordDataset(path).map(parse_func)
        dataset = dataset.shuffle(batch_size).batch(batch_size)
        return dataset
    
    def directory_info(self):
        """
        Print directory information.
        """
        pass

