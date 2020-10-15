from abc import ABC, abstractmethod
import tensorflow as tf
import os
# 일반적인 딥러닝 모델 학습을 위해 기본적으로 정의되어야 할 함수들을 정의해 놓은 추상화 클래스 입니다.
class Learning(ABC):
    @abstractmethod
    def __init__(self, args, database, network_cls): # args 누락된 것 수정
        self.args = args     # argument parser를 통해 받은 hyperparameter 집합

        # Get train/val/test dataset from database using self.get_*_dataset()
        self.database = database                      # type : tf.data.Dataset, shape : -
        self.train_dataset = self.get_train_dataset() # type : tf.data.Dataset, shape : -
        self.val_dataset = self.get_val_dataset()     # type : tf.data.Dataset, shape : -
        self.test_dataset = self.get_test_dataset()   # type : tf.data.Dataset, shape : -

        # General Learning setting
        '''
        self.networks_cls                   : Class of Omniglot or MiniImagenet Model(type : tf.keras.Model)
        self.model                          : Object of Omniglot or MiniImagenet Model(type : tf.keras.Model)

        self.learning_rate             : learning rate (in MAML, outer loop's learning rate)
        self.least_number_of_tasks_val_test : minimum number of tasks for validation & test
        self.clip_gradients                 : Applying gradient cliping or not 
        self.optimizer                      : optimizer 
        '''
        self.network_cls = networks_cls                                                  # type : tf.keras.Model Class
        self.model = self.network_cls(num_classes=self.n)                                # type : tf.keras.Model
        
        self.learning_rate = args.learning_rate                                          # type : float
        self.least_number_of_tasks_val_test = least_number_of_tasks_val_test             # type : int
        self.clip_gradients = clip_gradients                                             # type : bool
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) # type : tf.keras.optimizer.Adam()


        # Evaluation setting
        '''
        self.train_accuracy_metric : metric function to compute train accuracy
        self.train_loss_metric     : metric function to compute average train loss
        self.val_accuracy_metric   : metric function to compute validation accuracy
        self.val_loss_metric       : metric function to compute average validation loss
        '''
        self.train_accuracy_metric = tf.metrics.Accuracy() # type : tf.metrics.Accuracy object
        self.train_loss_metric = tf.metrics.Mean()         # type : tf.metrics.Mean object
        self.val_accuracy_metric = tf.metrics.Accuracy()   # type : tf.metrics.Accuracy object
        self.val_loss_metric = tf.metrics.Mean()           # type : tf.metrics.Mean object


        # Loging and Model Saving setting
        '''
        self.save_after_epochs                : Interval epoch for saving the model
        self.log_train_images_after_iteration : Interval iteration for writting the classified image on Tensorboard
        self.report_validation_frequency      : Interval epoch for reporting validation frequency

        self._root                            : Root path
        self.train_log_dir                    : Train log path 
        '''
        self.save_after_epochs = args.save_after_epochs                               # type : int
        self.log_train_images_after_iteration = args.log_train_images_after_iteration # type : int
        self.report_validation_frequency = args.report_validation_frequency           # type : int

        self._root = self.get_root()                                                         # type : string
        self.train_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/train/') # type : string

        try:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        except:
            createFolder(self.train_log_dir)
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_log_dir = os.path.join(self._root, self.get_config_info(), 'logs/val/')
        try:
            self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        except:
            createFolder(self.val_log_dir)
            self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)

        self.checkpoint_dir = os.path.join(self._root, self.get_config_info(), 'saved_models/')

    @abstractmethod
    def train(self):
        '''
        모델 Training을 수행하는 함수

        input : 
        - epochs (type : int, shape : -)

        output : None
        '''
        pass

    @abstractmethod
    def evaluate(self):

        '''
        Test set에 대하여 모델 Evaluation을 수행하는 함수
    
        input :
        - iterations (type : int, shape : -) : the number of gradient update in the inner loop
        - epochs_to_load_from (type : int, shape : -) : 몇 epoch를 학습한 모델을 불러올지 지정

        output : 
        - accuracy   (type : np.array, shape : -)
        '''
        pass

    @abstractmethod
    def predict(self):
        '''
        이미지 객체가 입력으로 들어가게 변경
        input data type : tensor : shape : (B, H, W, C) 
        이미지, 텍스트 든 뭐든 간에 입력 데이터를 받도록함
        경로를 받는다는 것은 load code가 있어야한다는 것이니 

        새로운 데이터(e.g. test set)를 입력받아 예측값을 출력하는 함수
        
        input : 
        predict_path : (type : string, shape : -) : 예측할 데이터셋의 상대경로 e.g. '/dataset/data/mini_imagenet/test'
        iterations : (type : int, shape : -) : the number of gradient update in the inner loop 
        epochs_to_load_from : (type : None, shape : -) : 몇 epoch를 학습한 모델을 불러올지 지정
        
        output : predicted value (type : tf.tensor, shape : (meta-test set size, ))
        '''
        pass

    @abstractmethod
    def get_train_dataset(self):
        '''
        train dataset을 불러오는 함수
        
        input : None

        output : train dataset (type : tf.data.dataset, shape : -)
        '''
        pass

    @abstractmethod
    def get_val_dataset(self):
        '''
        validation dataset을 불러오는 함수
        
        input : None

        output : validation dataset (type : tf.data.dataset, shape : -)
        '''
        pass

    @abstractmethod
    def get_test_dataset(self):
        '''
        validation dataset을 불러오는 함수
        
        input : None

        output : test dataset (type : tf.data.dataset, shape : -)
        '''
        pass

    @abstractmethod
    def get_config_info(self):
        '''
        model을 save할 때 기타 parameter 정보를 가져오는 함수

        input : None

        output : 'model-{모델명}_mbs-{meta batch size}_n-{N}_k-{K}_stp-{number of steps in meta learning} (type : string, shape : - )
        '''
        pass

# 메타 러닝 모델 학습을 위해 기본적으로 정의되어야 할 함수들을 정의해 놓은 추상화 클래스 입니다.
class MetaLearning(ABC):
    @abstractmethod
    def __init__(self,
                 args,
                 database,
                 network_cls,
                 least_number_of_tasks_val_test=-1,
                 # Make sure the validaiton and test dataset pick at least this many tasks.
                 clip_gradients=False
                 ):

        self.args = args
        # N-way K-shot setting
        ''''''
        self.n = args.n
        self.k = args.k
        self.meta_batch_size = args.meta_batch_size


        # Get train/val/test dataset from database using self.get_*_dataset()
        self.database = database                      # type : tf.data.Dataset, shape : -
        self.train_dataset = None   # type : tf.data.Dataset, shape : -
        self.val_dataset = None     # type : tf.data.Dataset, shape : -
        self.test_dataset = None    # type : tf.data.Dataset, shape : -

        # General Learning setting
        '''
        self.network_cls = network_cls
        self.model                          : Object of Omniglot or MiniImagenet Model(type : tf.keras.Model)

        self.meta_learning_rate             : learning rate (in MAML, outer loop's learning rate)
        self.least_number_of_tasks_val_test : minimum number of tasks for validation & test
        self.clip_gradients                 : Applying gradient cliping or not 
        self.optimizer                      : optimizer 
        '''
        self.network_cls = network_cls
        self.model = self.network_cls(num_classes=self.n) # type : tf.keras.Model
        
        self.least_number_of_tasks_val_test = least_number_of_tasks_val_test             # type : int
        self.meta_learning_rate = args.meta_learning_rate                                # type : float
        self.clip_gradients = clip_gradients                                             # type : bool
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.meta_learning_rate) # type : tf.keras.optimizer.Adam()

        # Evaluation setting
        '''
        self.train_accuracy_metric : metric function to compute train accuracy
        self.train_loss_metric     : metric function to compute average train loss
        self.val_accuracy_metric   : metric function to compute validation accuracy
        self.val_loss_metric       : metric function to compute average validation loss
        '''
        self.train_accuracy_metric = tf.metrics.Accuracy() # type : tf.metrics.Accuracy object
        self.train_loss_metric = tf.metrics.Mean()         # type : tf.metrics.Mean object
        self.val_accuracy_metric = tf.metrics.Accuracy()   # type : tf.metrics.Accuracy object
        self.val_loss_metric = tf.metrics.Mean()           # type : tf.metrics.Mean object




    @abstractmethod
    def get_root(self):
        '''
        모델이 실행되는 root path를 반환함
        checkpoint save를 위한 path 지정에 사용됨

        input : None

        output : os.path.dirname(__file__) : type : string
        '''
        pass

    @abstractmethod
    def meta_train(self, epochs):
        '''
        Meta-train을 수행하는 함수

        input : 
        - epochs (type : int, shape : -)

        output : None
        '''
        
        pass

    @abstractmethod
    def meta_test(self, iterations, epochs_to_load_from):
        '''
        Meta-test을 수행하는 함수
    
        input :
        - iterations (type : int, shape : -) : the number of gradient update in the inner loop
        - epochs_to_load_from (type : int, shape : -) : 몇 epoch를 학습한 모델을 불러올지 지정

        output : 
        - accuracy (type : np.array, shape : -)
        '''
        pass
    
    # 이거 MAML 도 바꿔야함
    @abstractmethod
    def predict_with_support(self, meta_test_path, iterations, epochs_to_load_from):
        '''
        새로운 데이터(e.g. test set)를 입력받아 예측값을 출력하는 함수
        
        input : 
        meta_test_path : (type : string, shape : -) : 예측할 데이터셋의 상대경로 e.g. '/dataset/data/mini_imagenet/test'
        iterations : (type : int, shape : -) : the number of gradient update in the inner loop 
        epochs_to_load_from : (type : None, shape : -) : 몇 epoch를 학습한 모델을 불러올지 지정
        
        output : predicted value (type : tf.tensor, shape : (meta-test set size, ))
        '''
        pass

    @abstractmethod
    def get_train_dataset(self):
        '''
        train dataset을 불러오는 함수
        
        input : None

        output : train dataset (type : tf.data.dataset, shape : -)
        '''
        pass

    @abstractmethod
    def get_val_dataset(self):
        '''
        validation dataset을 불러오는 함수
        
        input : None

        output : validation dataset (type : tf.data.dataset, shape : -)
        '''
        pass

    @abstractmethod
    def get_test_dataset(self):
        '''
        validation dataset을 불러오는 함수
        
        input : None

        output : test dataset (type : tf.data.dataset, shape : -)
        '''
        pass

    @abstractmethod
    def get_config_info(self):
        '''
        Log 기록시 파일명으로 사용할 hyper-parameter 정보를 불러오는 함수

        input : None

        output : 'model-{Omniglot or MiniImagenet}_mbs-{meta_batch_size}_n-{N}_k-{K}_stp-{number of steps in meta learning} (type : string, shape : - )
        '''
        pass