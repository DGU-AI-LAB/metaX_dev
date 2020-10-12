import tensorflow as tf
from tensorflow.keras import layers,optimizers,losses
from dataset.data.oxfordflower.tf_utils import * # 20.10.08. Modifyed for path matching
from model.LearningType import Learning
from dataset.data.oxfordflower.data_utils import parse_config # 20.10.08. Modifyed for path matching
import os
import pickle

class PretrainModel(tf.keras.Model):
    """
    Just a wrapper model for a model in tf.keras.applications, adjusting some layers suitablefor modified mcnn.
    In order to be applicable to different models, pool_size_checker or dim_checker can be changed
    
    save_by_ckpt(load_by_ckpt) is to save(load) the model 1. without the variables related to a optimizer 
    2.Save easily masking a certain set of layers
    """
    def __init__(self,encoder_name="MobileNetV2",model_name="Modified_m_CNN",is_pred=True):

        super(PretrainModel,self).__init__()

        pool_size = pool_size_checker(encoder_name)
        last_vec_dim = dim_checker(encoder_name)
    
        self.is_pred = is_pred
        
        model_func = getattr(tf.keras.applications,encoder_name)
        
        self._model = model_func(include_top=False,input_shape=(224,224,3))
        self.pool2d = layers.MaxPool2D(pool_size)
        self.fcVec = layers.Dense(last_vec_dim)
        
        if self.is_pred:
            self.fc = layers.Dense(102)
            
        self.flatten = layers.Flatten()
        
    def call(self,inp):
        out = self._model(inp)
        out = self.pool2d(out)
        out = self.fcVec(out)
        if self.is_pred:
            out = self.fc(out)
        out = self.flatten(out)
        return out

    def save_by_ckpt(self,save_path,mask_layers=[3]):
        _dict = {}
        offset = 0
        for i,layer in enumerate(self.layers):
            if i in mask_layers:
                offset += 1
                continue
            _dict[str(i-offset)] = layer
        path = tf.train.Checkpoint(**_dict).save(save_path)
        return path,_dict
    
    def load_by_ckpt(self,saved_path,is_assert=False):
        _dict_re={}
        for i,layer in enumerate(self.layers):
            _dict_re[str(i)] = layer
        restore_status = tf.train.Checkpoint(**_dict_re).restore(saved_path)
        if is_assert:
            restore_status.assert_consumed()

class conv_block(tf.keras.layers.Layer):
    """
    Conv, (optional)BatchNormal, Dropout, Maxpool2d
    """
    def __init__(self,num_filters, kernel_size, rate, max_seq=None, is_bn=True):
        super(conv_block,self).__init__()
        _poolsize =  [max_seq - kernel_size[0] + 1,1] if max_seq is not None else [2,1]
        _stride= [1,1] if max_seq is not None else None

        self.conv = layers.Conv2D(num_filters, kernel_size=kernel_size, padding='valid', kernel_initializer='he_normal', activation='relu')
        self.bn =  layers.BatchNormalization()
        self.drop_layer = layers.Dropout(rate)
        self.pool2d = layers.MaxPool2D(pool_size=_poolsize,strides=_stride,padding="valid")
        self.is_bn = is_bn
        
    def call(self,inp,training=True):
        out=self.conv(inp)
        if self.is_bn :
            out= self.bn(out,training)
        out = self.drop_layer(out,training)
        return self.pool2d(out)

class Modified_m_CNN(tf.keras.Model):
    """
    Modified_m_CNN model, implemented with tf.keras.Model
    """    
    def __init__(self,config,save_path,encoder_name,preset_trainable):

        super(Modified_m_CNN,self).__init__()
        self.config = config
        
        LAMBDA=self.config["lambda"]
        DROP_OUT=self.config["drop_out"]
        DROP_OUT2=self.config["drop_out2"]
        dropouts = [DROP_OUT,DROP_OUT2]
        kernel_sizes=self.config["filter_sizes"]
        num_filters = self.config["num_filters"]
        
        if "vocab_size" not in config:
            with open(os.path.join(config["data_path"],config["vocab"]),"rb") as f:
                self.vocab_size = len(pickle.load(f))
        else:
            self.vocab_size = config["vocab_size"]
        
        self.preset_trainable = preset_trainable
        self.preset_model = self.encoder_loader(save_path,encoder_name,is_trainable=preset_trainable)
        self.bn = layers.BatchNormalization()
        self.conv_blockx = conv_block(256,(14,1),dropouts[1],None,is_bn=False)
        self.conv_block0 = conv_block(num_filters,(kernel_sizes[0], config["embed_dim"]),dropouts[1],config["max_len"])
        self.conv_block1 = conv_block(num_filters,(kernel_sizes[1], config["embed_dim"]),dropouts[1],config["max_len"])
        self.conv_block2 = conv_block(num_filters,(kernel_sizes[2], config["embed_dim"]),dropouts[1],config["max_len"])
        self.conv_block3 = conv_block(512,(5,1),dropouts[0],None)
        self.fc = layers.Dense(config["num_class"],kernel_initializer='he_normal')
        
        embedding_path = os.path.join(config["pretrain_path"] ,config["embedding"])
        if os.path.exists(embedding_path):
            print("\nPretrained skipgram,Embedding matrix loaded, from {}\n".format(embedding_path))
            embedding_array = np.load(embedding_path).astype(np.float32)
            initializer = tf.keras.initializers.Constant(embedding_array)
        else:
            initializer = "uniform"
        self.embed = layers.Embedding(self.vocab_size,config["embed_dim"],
                            input_length=config["max_len"],embeddings_initializer=initializer)
                            
        self.dropout_layer = layers.Dropout(dropouts[0])
        
        self.flatten = layers.Flatten()
        
    def call(self,inp,training=True):
        img, txt = inp["img"],inp["txt"] #inp["image"],inp["text"]
        img_vec = self.preset_model(img)
        # Image through vgg
        img_vec = tf.reshape(img_vec,[-1,16,1,256])
        img_vec = self.bn(img_vec,training)
        conv_x = self.conv_blockx(img_vec)
        
        # Text through embedding
        txt_embedded = self.embed(txt)
        txt_embeddded =  tf.expand_dims(txt_embedded,3)
        #tf.reshape(txt_embedded,[-1,config["max_len"],config["embed_dim"],1])
        conv_0 = self.conv_block0(txt_embeddded,training)
        conv_1 = self.conv_block1(txt_embeddded,training)
        conv_2 = self.conv_block2(txt_embeddded,training)
        
        concat1 = tf.concat([conv_0,conv_x],axis=1)
        concat2 = tf.concat([conv_1,conv_x],axis=1)
        concat3 = tf.concat([conv_2,conv_x],axis=1)
        
        concat_total = tf.concat([concat1,concat2,concat3],axis=1)
        conv_total = self.conv_block3(concat_total,training)
        #conv_total = tf.squeeze(conv_total)
        conv_total = self.dropout_layer(conv_total,training)
        
        output = self.fc(conv_total)
        output = self.flatten(output)
        return output
    
    @staticmethod
    def encoder_loader(save_path=None,encoder_name=None,is_trainable=False):
        assert encoder_name is not None, "Please specifiy image encoder model, one of tf.keras.applications"
        if save_path is None:
            print("\nGet preset model, {}, trained with ImageNet\n".format(encoder_name))
            _model = PretrainModel(encoder_name,is_pred=False)
        else:
            _model = PretrainModel(encoder_name,is_pred=False)
            _model.load_by_ckpt(save_path)
            print("\nGet preset model, {}, trained with Target Data, from {}\n".format(encoder_name,save_path))
            
        _model.trainable=is_trainable
        
        return _model

class Hetero(Learning):
    """
    Take Model(keras.Model) and Database, and pretrain an Image encoder part of the Model
        
    Argument
        args : ArgumentParse (encoder_name , batch_size, epochs, etc)
        database : one of subclass of Database
        network_cls : tf.keras.Model(Modified_m_CNN or VQA)￣s
        #encoder_name : (string)image classfication model, one of tf.keras.applications
    """
    
    def __init__(self,args,config_path,database,network_cls):
       
        self.config = parse_config(config_path)
        config = self.config
        self.args = args
        self.database = database
        self.network_cls = network_cls
        
        # Initialize an optimizer and an loss func
        
        if self.config["clip_norm"]==.0:
            opt_dict = {"learning_rate":self.config["lr"]}
        elif self.config["clip_norm"]>.0:
            opt_dict = {"learning_rate":self.config["lr"],"clipnorm":self.config["clip_norm"]}
        else:
            raise ValueError("clip_norm should be 0(No clipping) or greater float")
        
        self.optimizer = getattr(optimizers, args.optimizer)(**opt_dict)
        
        if self.args.binary:
            self.loss_func = losses.BinaryCrossentropy(from_logits=True)
        else:
            self.loss_func = losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # @ Log and model file path
        # 3세부에서 UI 출력 화면을 위한 데이터를 쉽게 받아올 수 있도록
        # 'dataset/data/ui_output/mcnn/' 내에 Step 별로 저장하도록 수정 부탁드립니다.
        # 아래는 예시를 위해 csv log path만 수정해보았습니다.
        # e.g. 학습 기록 출력 - Train Step -> mcnn/step3/log.csv
        # self.csv_log_cb = tf.keras.callbacks.CSVLogger(
        #     os.path.join('dataset/data/ui_output/mcnn/',config["csv_path"],"log.csv"))
ㄴ
        # @ Changed path to dataset/data/ui_output/mcnn/
        self.ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config["model_path"],config["log_path"],config["ckpt_path"],f"{args.encoder_name}.ckpt"))
        self.csv_log_cb = tf.keras.callbacks.CSVLogger(
            os.path.join(config["model_path"],config["log_path"],config["csv_path"],"log.csv"))
        self.tb_cb = tf.keras.callbacks.TensorBoard(
            os.path.join(config["model_path"],config["log_path"],config["tb_path"]))    
        super(Hetero,self).__init__(args) # 아직 구현하지 못했는데, 여러상황들 고려해서 다시 구성할 예정입니다.
        
    def get_train_dataset(self):
        return self.database.data_loader("train","both",self.args.batch_size,self.args.batch_size*10)

    def get_val_dataset(self):
        return self.database.data_loader("eval","both",self.args.batch_size,self.args.batch_size*10)

    def get_test_dataset(self):
        return self.database.data_loader("test","both",self.args.batch_size,self.args.batch_size*10)
        
    def train(self):
        config = self.config
        
        if self.args.pretrain:
            path,_dict = self.pretrain_img_model(
                config,encoder_name=self.args.encoder_name,batch_size=self.args.batch_size2,
                epochs=self.args.epochs2,loss=self.loss_func,optimizer=self.args.optimizer,
                metrics=["accuracy"])
        else:
            path = None
        
        trn_dataset = self.get_train_dataset()
        eval_dataset = self.get_val_dataset()
    
        self.model = self.network_cls(config,path,self.args.encoder_name,preset_trainable=self.args.fineTune)
    
        if not os.path.exists(os.path.join(config["model_path"],config["log_path"],config["csv_path"])):
            os.path.makedirs(os.path.join(config["model_path"],config["log_path"],config["csv_path"]))
        
        self.model.compile(loss =self.loss_func,optimizer=self.optimizer,metrics=["accuracy"])
        self.model.fit(trn_dataset,epochs=self.args.epochs,
            callbacks=[self.ckpt_cb,self.csv_log_cb,self.tb_cb],validation_data = eval_dataset)
        
    def predict(self,inp):
        """
        inp should be a dictionary, which is {"img":image_tensor,"txt":text_token_tensor}
        """
        return self.model(inp)
    
    def evaluate(self): 
        """
        Get accuracy and loss for test dataset
        """
        test_dataset = self.get_test_dataset()
        # Load from checkpoint is required
        self.model.evaluate(test_dataset)
    
    def pretrain_img_model(self,config,usage="train",batch_size=64,epochs=30,**kwargs):
        """
        Pretrain Image encoder as a set of arguments
        """
        encoder_name = self.args.encoder_name
        tf_path = os.path.join(config["data_path"],config["tfrecord_path"],usage+".record")
                
        dataset = self.database.data_loader("train","image",self.args.batch_size2,self.args.batch_size2*10)
        eval_dataset = self.database.data_loader("eval","image",self.args.batch_size2,self.args.batch_size2*10)
        
        preset_model = PretrainModel(encoder_name=encoder_name,is_pred=True)
        preset_model.compile(**kwargs)
        preset_model.fit(dataset,epochs=epochs,validation_data = eval_dataset)
        

        # 학습된 모델이 저장되는 부분을 
        # dataset/data/ui_output/[모델명]/[모델이 저장되는 Step]
        # 으로 통일하고자 합니다.
        # 예를 들어 step3가 model training 이면 아래 경로에 학습한 모델 및 pretrained model을 저장합니다.
        # e.g. dataset/data/ui_output/mcnn/step3/  
         
        path,_dict = preset_model.save_by_ckpt(
            os.path.join(config["model_path"],config["pretrain_path"],encoder_name,"ck.ckpt"),
            mask_layers=[3]) #mask_layers should be determined more flexibly, for example, mask_layers can be set
            
        return path,_dict

    def get_config_info(self):
        whole_dict = {}
        whole_dict.update(
            dict(self.args._get_kwargs())
            )
        whole_dict.update(self.config)
        
        return_keys = ("encoder_name","model_name","batch_size","max_len","num_class","img_size","embed_dim","optimizer")
        
        return {key:val for key,val in whole_dict.items() if key in return_keys}

""" Train.py을 모방
import sys # del sys.modules["model.hetero.mcnn"]
from model.hetero.mcnn import *
from dataset.oxford_generator import OxfordFlower
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--batch_size2",type=int,default=64)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--epochs2",type=int,default=5)

parser.add_argument("--encoder_name",type=str,default="MobileNetV2")
parser.add_argument("--optimizer",type=str,default="SGD")

parser.add_argument("--binary",type=bool,default=False)
parser.add_argument("--pretrain",type=bool,default=True)
parser.add_argument("--fineTune",type=bool,default=True)

args = parser.parse_args()

config_path = "dataset/oxfordflower/args.ini"
database = OxfordFlower(config_path)

if True: # args.model_name == "Modified_m_CNN"
    network_cls = Modified_m_CNN

hetero_inst = Hetero(args,database,network_cls)
hetero_inst.get_config_info()

hetero_inst.train()
hetero_inst.evaluate()
"""
   
