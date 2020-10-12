import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from utils import combine_first_two_axes, createFolder
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, BatchNormalization

from dataset.data.MSCOCOKR_data_generator import MSCOCOKRDatabase
from model.LearningType import Learning
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



class ImageCaptioningModel(Learning):

    # @ network_cls 인자로 자기자신을 입력받기만 하고 실제로는 쓰지 않습니다.
    def __init__(self, args, database, network_cls):
        
        super(ImageCaptioningModel, self).__init__(
            args,
            database,
            network_cls
            )
        
        self.max_length = 38
        # @ vocab_size를 설정해주시기 바랍니다. 코드가 동작하지 않습니다.
        self.vocab_size = ???
        self.tokenizer = database.load_tokenizer()
        # @ 아래 두 path의 text file이 없습니다. 
        self.clean_descriptions_path = database.train_address + "/ms_coco_2014_kr_train_token_clean.txt"
        self.image_filename_path = database.train_address + "/ms_coco_2014_kr_train_images.txt"
        self.train_imgs = database.load_photos(image_filename_path)
        self.train_descriptions = database.load_clean_descriptions(clean_descriptions_path, train_imgs) 
        self.train_features = database.load_feature()
        
        self.epochs = args.epochs
        self.iterations = args.iterations

    # @ ??? 의 값을 지정해줄 것
    def train(self, epoch=10, iterations=???):
        model = self.define_model(self.vocab_size, self.max_length)
        
        for i in tqdm(range(epochs), desc = "train epoch"):
            generator = self.get_train_dataset()
            model.fit_generator(generator, epochs=1, steps_per_epoch= iterations, verbose=1)
        
        model.save("dataset/data/MSCOCOKR_data/train/ImagecaptioningModel.h5")
        

    def get_train_dataset(self):
        generator = self.data_generator(self.train_descriptions, self.train_features, self.tokenizer, self.max_length)
        return generator
        
    def get_val_dataset(self):
        pass
        
    def get_test_dataset(self):
        pass
        
    def get_config_info(self):
        model = self.define_model(self.vocab_size, self.max_length)
        print(model.summary())
        
    def evaluate(self):
        test_image_path = "dataset/data/MSCOCOKR_data/test/test_images"
        token_split = pd.read_csv("dataset/data/MSCOCOKR_data/test/token_dataframe.csv', encoding='cp949')
        
        inception_model = Xception(include_top=False, pooling="avg")
        test_image_name = os.listdir(test_image_path)
        
        test_image_full_name = []
        for i in range(len(test_image_name)):
            tmp = test_image_path + '/' + test_image_name[i]
            test_image_full_name.append(tmp)
        
        bleu_all=[]
        max_length= self.max_lenght
        # @ load_model 을 별도의 메서드로 만들어 주시기 바랍니다(라이브러리 형식 맞추기위함)
        model = load_model("dataset/data/MSCOCOKR_data/train/ImagecaptioningModel.h5")
        
        for i in tqdm(range(len(test_image_full_name)), desc = "test_image_caption"):
            img_path = test_image_full_name[i]
            
            photo = self.extract_features(img_path, inception_model)
            
            tokenizer = self.tokenizer
            description = self.generate_desc(model, tokenizer, photo, max_length)
            description=description[6:-3] # removing start and end.
            
            #테스트 이미지에 대한 캡션들 가져오기
            test_image_name = os.listdir(test_image_path)[i]
            reference = list(token_split.loc[token_split['image_id'] == test_image_name]['caption'])
            
            #BLEU 계산을 위한 문장 split
            reference = [s.split(' ') for s in reference]
            candidate = description.split(' ')
         
            #BLEU 계산
            bleu_1gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            bleu_2gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
            bleu_3gram = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu_4gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            
            bleu = [bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram]
            bleu_all.append(bleu)
        
        #BLEU 데이터프레임 생성    
        bleu_all = pd.DataFrame(bleu_all)
        bleu_all.columns = ['bleu_1','bleu_2','bleu_3','bleu_4']
        # bleu_mean = bleu_all.mean(axis=0)  
        
        bleu_1_mean = bleu_all["bleu_1"].mean()
        bleu_2_mean = bleu_all["bleu_2"].mean()
        bleu_3_mean = bleu_all["bleu_3"].mean()
        bleu_4_mean = bleu_all["bleu_4"].mean()
    
        print("BLEU 1 : ", bleu_1_mean, "\n",
              "BLEU 2 : ", bleu_2_mean, "\n",
              "BLEU 3 : ", bleu_3_mean, "\n",
              "BLEU 4 : ", bleu_4_mean)
        
    # @ test_image_path를 train.py로 빼고, 이 메서드의 인자로 받게 해주시기 바랍니다.
    def predict(self):
        #폰트 경로
        font_path = "dataset/data/MSCOCOKR_data/test/NanumGothicLight.TTF"
        
        #폰트 이름 얻어오기
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        
        #font 설정
        matplotlib.rc('font',family=font_name)
        
        
        
        test_image_path = "dataset/data/MSCOCOKR_data/test/test_images"

        test_image_name = os.listdir(test_image_path)
        
        test_image_full_name = []
        for i in range(len(test_image_name)):
            tmp = test_image_path + '/' + test_image_name[i]
            test_image_full_name.append(tmp)
        
        predict_description=[]
        max_length = self.max_lenght
                
        inception_model = Xception(include_top=False, pooling="avg")
        
        model = load_model("dataset/data/MSCOCOKR_data/train/ImagecaptioningModel.h5")
        
        result_path = "dataset/data/MSCOCOKR_data/test/test_images_result"
        
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        
        for i in tqdm(range(test_image_full_name), desc = "test_image_caption"):
            img_path = test_image_full_name[i]
            
            photo = self.extract_features(img_path, inception_model)

            tokenizer = self.tokenizer
            
            img = Image.open(img_path)
            description = self.generate_desc(model, tokenizer, photo, max_length)
            description=description[6:-3] # removing start and end.
            
            #예측 캡션 객체 생성
            predict_description.append(description)
            
            #결과 이미지 저장
            plt.imshow(img)
            plt.title(description)
            plt.savefig(result_path + "/" + test_image_name[i])
            plt.close()        
            
        return predict_description

    
    def extract_features(self, filename, model):
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
    
    # @ 이걸 위의 OmniglotModel처럼 밖으로 빼주시면 됩니다.
    #   그리고 train.py에서 이걸 network_cls로 받아 주시면 됩니다.
    def define_model(self, vocab_size, max_length):
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(512, activation='relu')(fe1) # 256, 512, 1024
       
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = GRU(512)(se2) # GRU
    
        decoder1 = concatenate([fe2, se3]) # add or concatenate
        decoder2 = Dense(512, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        return model
    
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
        
    