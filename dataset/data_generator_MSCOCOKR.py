from abc import ABC
from PIL import Image
from pickle import load
from tqdm import tqdm
import os # @ package added

class MSCOCOKRDatabase(ABC):

    def __init__(self, train_address, test_address):
        # @ 상대경로 -> 절대경로 수정
        self.train_address = os.path.join(os.getcwd(), train_address)
        self.test_address = os.path.join(os.getcwd(), test_address)
      
    def preview_image(self, image_path):
        image = Image.open(image_path)
        image.show()
        return image
    
    def get_input_shape(self):
        return 299, 299, 3

    def load_tokenizer(self):
        tokenizer_path = self.train_address + "/tokenizer.p"
        tokenizer = load(open(tokenizer_path,"rb"))
        return tokenizer
    
    def load_train_feature(self):
        features_path = self.train_address + "/features.p"
        features = load(open(features_path,"rb"))
        return features
    
    def load_clean_descriptions(self, filename, photos): 
        file = self.load_doc(filename)
        descriptions = {}
        for line in tqdm(file.split("\n"), desc = "load clean description"):
            words = line.split()
            if len(words)<1 :
                continue
            image, image_caption = words[0], words[1:]
            if image in photos:
                if image not in descriptions:
                    descriptions[image] = []
                desc = '<start> ' + " ".join(image_caption) + ' <end>'
                descriptions[image].append(desc)
        return descriptions
    
    def load_doc(self, filename):
        file = open(filename, 'r')
        text = file.read()
        file.close()
        return text
    
    def load_photos(self, filename):
        file = self.load_doc(filename)
        photos = file.split("\n")[:-1]
        return photos
    # @ 20.10.15. 2세부 추가 발송본 수정 부분
    def load_feature(self, photos, feature_path):
        #loading all features
        all_features = load(open(feature_path,"rb"))
        #selecting only needed features
        features = {k:all_features[k] for k in tqdm(photos, desc = "load photo features")}
        return features

