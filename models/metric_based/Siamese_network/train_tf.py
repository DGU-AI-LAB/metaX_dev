import tensorflow as tf
import time
from Siamese_network_tf import SiameseNetwork
from data_generator_tf import DataGenerator

train_dataset = DataGenerator(image_size = (105, 105, 1), batch_size=10, N = 2, K = 2)
test_dataset = DataGenerator(image_size = (105, 105, 1), batch_size=10, N = 5, K = 5)

print("data prepared")

model = SiameseNetwork()
 
print("model prepared")

model_save_dir = "./models_tf/before_train_{}".format(int(time.time()))
model.save_weights(model_save_dir)

print("model saved before train, train start")

model.train_model(train_dataset = train_dataset, num_epoch = 10000, learning_rate = 0.05)
model_save_dir = "./models_tf/after_train{}".format(int(time.time()))
model.save_weights(model_save_dir)

print("model saved after train")

# print(model.test_model(test_dataset, num_epoch = 100))