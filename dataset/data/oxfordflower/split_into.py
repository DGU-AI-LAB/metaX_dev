import numpy as np
import json
import os,sys
import random

imgs = os.listdir("images")
txts = os.listdir("texts")

for img in imgs:
    _, num = int(img.split("_")[-1])
    

match_dict = { }
