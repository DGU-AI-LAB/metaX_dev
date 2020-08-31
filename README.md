# metaX_Library
<img src="https://github.com/DGU-AI-LAB/metaX_dev/blob/master/logo_transparent.png" width="300">


metaX library is a python library with deep neural networks and datasets for meta learning and multi-view learning base on Tensorflow 2.0.

We provide...
- Deep neural networks for meta learning that can solve one-shot or few-shot problems.
- Deep neural networks for multi-view learning
- Various types of experimental datasets that can be used for experiments using the provided model 
- Load codes for provided dataset with few-shot learning settings


## Directory
<pre>
<code>
dataset/
	data_generator.py (Omniglot, mini-ImageNet) (Completed)    
        KTS_data_generator.py                       (Completed)                    
	FLOWER_data_generator.py                    (In progress)             
	KMSCOCO_data_generator.py                   (In progress)           
	KVQA_data_generator.py                      (In progress)              
	CropDisease.py                              (Completed)
	EuroSAT.py                                  (Completed) 
	ISIC.py                                     (Completed)   
 	ChestX.py                                   (Completed)
  data/
  raw_data/
  
model/
	LearningType.py 
	metric_based/
		Relation_network.py                 (In progress)
		Prototypical_network.py             (In progress)
		Siamese_network.py                  (Completed)
	model_based/
		MANN.py                             (Completed)
		SNAIL.py
	optimization_based/
		MAML.py                             (Completed)
		MetaSGD.py
		Reptile.py                          (In progress)
	heterogeneous_data_analysis/
		image_text_embeding.py              (In progress)
		Vis_LSTM.py                         (In progress)
    Modified_mCNN.py                                (In progress)
train.py
utils.py (accuracy, mse)
