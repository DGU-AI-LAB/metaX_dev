import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def pool_size_checker(encoder_name,model_name="Modified_m_CNN"):
    if encoder_name in ["InceptionV3","VGG16"]:
        pool_size = (5,5)
    elif encoder_name in ["MobileNetV2"]:
        pool_size = (7,7)
    return pool_size

def dim_checker(encoder_name,model_name="Modified_m_CNN"):
    return 4096
    

def config_wrapper_parse_funcs(config):
    
    description={
        "img_raw":tf.io.FixedLenFeature([],tf.string),
        "text_ids":tf.io.FixedLenFeature([config["max_len"]],tf.int64),
        "label":tf.io.FixedLenFeature([],tf.int64),
        "id":tf.io.FixedLenFeature([],tf.int64)
    }
    
    def _parse_img_example(example_proto):
        parsed_feature=tf.io.parse_single_example(example_proto,description)
        image = parsed_feature["img_raw"]
        image = tf.io.decode_raw(image,tf.uint8)
        image = tf.cast(image,tf.float32)
        image = tf.reshape(image,config["img_size"])
        image = tf.image.per_image_standardization(image)  
        
        label = parsed_feature["label"]

        return image,label
    
    def _parse_txt_example(example_proto):
        parsed_feature=tf.io.parse_single_example(example_proto,description)
        text_ids = parsed_feature["text_ids"]
        
        label = parsed_feature["label"]

        return text_ids,label 
    
    def _parse_single_example(example_proto):
        parsed_feature=tf.io.parse_single_example(example_proto,description)
        image = parsed_feature["img_raw"]
        image = tf.io.decode_raw(image,tf.uint8)
        image = tf.cast(image,tf.float32)
        image = tf.reshape(image,config["img_size"])
        image = tf.image.per_image_standardization(image)  
        
        text_ids = parsed_feature["text_ids"]
        
        label = parsed_feature["label"]

        return ({"img":image, "txt": text_ids},label)
    
    return _parse_img_example,_parse_txt_example,_parse_single_example
