import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization

from dataset.data_generator_MSCOCOKR import MSCOCOKRDatabase
from model.LearningType import Learning
# TODO : modify the code form Like this for PyPi Build
# from metaX.dataset.data_generator_MSCOCOKR import MSCOCOKRDatabase
# from metaX.model.LearningType import Learning
from keras.preprocessing.text import Tokenizer
from PIL import Image
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, GRU
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
import re
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from keras.applications.xception import Xception
from matplotlib import font_manager,rc
import matplotlib
import matplotlib.pyplot as plt
from pickle import dump, load
from keras import optimizers

class MSCOCOKRModel(tf.keras.Model):
    
    def define_model(vocab_size, max_length, Embedding_size=512):
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(512, activation='relu')(fe1) # 256, 512, 1024
       
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, Embedding_size, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = GRU(512)(se2) # GRU
    
        decoder1 = concatenate([fe2, se3]) # add or concatenate
        decoder2 = Dense(512, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

        
        return model


class ImageCaptioningModel(Learning):
    
    def __init__(self, args, database, network_cls):
        self.train_path = database.train_address
        self.val_path = database.val_address
        self.test_path = database.test_address
        # CAUTION : Dept2 Modified LearningType.py
        database.prepare_database()

        super(ImageCaptioningModel, self).__init__(
            args,
            database,
            network_cls
            )
        
        self.train_image_path = os.path.join(database.train_address, "train_images")
        clean_descriptions_path = os.path.join(database.train_address, "ms_coco_2014_kr_train_token_clean.txt")
        image_filename_path = os.path.join(database.train_address,"ms_coco_2014_kr_train_images.txt")
        train_imgs = database.load_photos(image_filename_path)
        self.train_descriptions = database.load_clean_descriptions(clean_descriptions_path, train_imgs) 
        self.train_features = database.extract_features(self.train_image_path)
        self.max_length = database.max_length(self.train_descriptions)
        self.tokenizer = database.create_tokenizer(self.train_descriptions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.epochs = args.epochs
        self.iterations = args.iterations
        self.Embedding_size = args.Embedding_size        
       
    def get_train_dataset(self):
        # TODO : This file doesn't exist
        filename =  os.path.join(self.train_path, "ms_coco_2014_kr_train_token_clean.txt")
        file = self.load_doc(filename)
        captions = file.split('\n')
        descriptions ={}
        for caption in captions[:-1]:
            if len(caption.split('\t')) > 1:
                img, caption = caption.split('\t')
                if img[:-2] not in descriptions:
                    descriptions[img[:-2]] = [caption]
                else:
                    descriptions[img[:-2]].append(caption)
        return descriptions
                        
    def get_val_dataset(self):
        filename =  os.path.join(self.val_path, "ms_coco_2014_kr_val_token_clean.txt")
        file = self.load_doc(filename)
        captions = file.split('\n')
        descriptions ={}
        for caption in captions[:-1]:
            if len(caption.split('\t')) > 1:
                img, caption = caption.split('\t')
                if img[:-2] not in descriptions:
                    descriptions[img[:-2]] = [caption]
                else:
                    descriptions[img[:-2]].append(caption)
        return descriptions
        
    def get_test_dataset(self):
        filename =  os.path.join(self.test_path, "ms_coco_2014_test_imgid.txt")
        file = self.load_doc(filename)
        captions = file.split('\n')
        descriptions ={}
        for caption in captions[:-1]:
            if len(caption.split('\t')) > 1:
                img, caption = caption.split('\t')
                if img[:-2] not in descriptions:
                    descriptions[img[:-2]] = [caption]
                else:
                    descriptions[img[:-2]].append(caption)
        return descriptions


    def train(self):
        iterations=len(self.train_descriptions)
        # train_image_path = os.path.join(self.train_path, "train_images")
        self.model = MSCOCOKRModel.define_model(
            self.vocab_size, self.max_length, self.Embedding_size)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.Adam(lr=self.learning_rate))
        # self.model.summary()
        
        for i in range(self.epochs):
            generator = self.data_generator(self.train_descriptions, self.train_features, self.tokenizer, self.max_length)
            self.model.fit_generator(generator, epochs=1, steps_per_epoch= iterations, verbose=1)
        
        # TODO : This code should be form unified with mccn.py
        save_path = os.path.join(self.train_path, "ImagecaptioningModel.h5")
        self.model.save(save_path)
        
    def load_doc(self, filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text

    def get_config_info(self):
        model = self.define_model(self.vocab_size, self.max_length)
        print(model.summary())
        
    def evaluate(self):
        val_path = self.val_path
        val_image_path = os.path.join(val_path, "val_images")
        token_split_path = os.path.join(val_path, "token_dataframe.csv")
        token_split = pd.read_csv(token_split_path, encoding='cp949')
        
        inception_model = Xception(include_top=False, pooling="avg")
        val_image_name = os.listdir(val_image_path)
        
        val_image_full_name = []
        for i in range(len(val_image_name)):
            tmp = os.path.join(val_image_path, val_image_name[i])
            val_image_full_name.append(tmp)
        
        bleu_all=[]
        max_length= self.max_length

        # Overwrite save model with self.model 
        save_path = os.path.join(self.train_path, "ImagecaptioningModel.h5")
        self.model = load_model(save_path)
        
        for i in tqdm(range(len(val_image_full_name)), desc = "val_image_caption"):
            img_path = val_image_full_name[i]
            
            photo = self.extract_features_test(img_path, inception_model)
            
            tokenizer = self.tokenizer
            description = self.generate_desc(model, tokenizer, photo, max_length)
            description=description[6:-3] # removing start and end.
            
            #테스트 이미지에 대한 캡션들 가져오기
            val_image_name = os.listdir(val_image_path)[i]
            reference = list(token_split.loc[token_split['image_id'] == val_image_name]['caption'])
            
            #BLEU 계산을 위한 문장 split
            reference = [s.split(' ') for s in reference]
            candidate = description.split(' ')
            
            #BLEU 계산        
            bleu_1gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_2gram = sentence_bleu(reference, candidate, weights=(0.8, 0.2, 0, 0))
            bleu_3gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_4gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))            
            bleu = [bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram]
            bleu_all.append(bleu)
        
        #BLEU 데이터프레임 생성    
        bleu_all = pd.DataFrame(bleu_all)
        bleu_all.columns = ['bleu_1','bleu_2','bleu_3','bleu_4']
        # bleu_all.to_csv(self.val_path + '/bleu_score.csv')
        # bleu_mean = bleu_all.mean(axis=0)  
        
        bleu_1_mean = bleu_all["bleu_1"].mean()
        bleu_2_mean = bleu_all["bleu_2"].mean()
        bleu_3_mean = bleu_all["bleu_3"].mean()
        bleu_4_mean = bleu_all["bleu_4"].mean()
    
        print(" BLEU 1 : ", bleu_1_mean, "\n",
              "BLEU 2 : ", bleu_2_mean, "\n",
              "BLEU 3 : ", bleu_3_mean, "\n",
              "BLEU 4 : ", bleu_4_mean)
        
        
    def predict(self):
        
        # test_path = self.test_path
        # 폰트 경로
        font_path = os.path.join(self.test_path, "NanumMyeongjo.TTF")
        # font_path = "C:/Windows/Fonts/batang.ttc"
        
        #폰트 이름 얻어오기
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        #font 설정
        matplotlib.rc('font',family=font_name)
        
        test_image_path = os.path.join(self.test_path, "test_images")

        test_image_name = os.listdir(test_image_path)
        
        test_image_full_name = []
        for i in range(len(test_image_name)):
            tmp = test_image_path + '/' + test_image_name[i]
            test_image_full_name.append(tmp)
        
        predict_description=[]
        max_length = self.max_length
                
        inception_model = Xception(include_top=False, pooling="avg")
        
        save_path = os.path.join(self.train_path, "ImagecaptioningModel.h5")
        # Overwrite save model with self.model 
        self.model = load_model(save_path)
        
        result_path = os.path.join(self.test_path, "test_images_result")
        
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        
        for i in tqdm(range(len(test_image_full_name)), desc = "test_image_caption"):
            img_path = test_image_full_name[i]
            
            photo = self.extract_features_test(img_path, inception_model)

            tokenizer = self.tokenizer
            
            img = Image.open(img_path)
            description = self.generate_desc(self.model, tokenizer, photo, max_length)
            description=description[6:-3] # removing start and end.
            
            #예측 캡션 객체 생성
            predict_description.append(description)
            
            #결과 이미지 저장
            plt.imshow(img)
            plt.title(description)
            plt.savefig(result_path + "/" + test_image_name[i])
            plt.close()        
            
        return predict_description

    
    def extract_features_test(self, filename, model):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)

        image = np.expand_dims(image, axis=0)
        image_shape_temp = image.shape
        if len(image_shape_temp) < 4: # 흑백사진 처리
            image = np.expand_dims(image, axis=3)
            image = np.append(image, np.append(image, image, axis = 3), axis = 3)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
            
    
                
    def create_sequences(self, tokenizer, max_length, desc_list, feature):
        X1, X2, y = list(), list(), list()
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                # store
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)
    
    
    def data_generator(self, descriptions, features, tokenizer, max_length):
        while 1:
            for key, description_list in descriptions.items():
                #retrieve photo features
                feature = features[key][0]
                input_image, input_sequence, output_word = self.create_sequences(tokenizer, max_length, description_list, feature)
                yield [[input_image, input_sequence], output_word]
    

    
    def word_for_id(self, integer, tokenizer):
      for word, index in tokenizer.word_index.items():
          if index == integer:
              return word
          
    def generate_desc(self, model, tokenizer, photo, max_length):
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = self.word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text
        
    