import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


class df_to_tfrec:
    
    def __init__(self,df, output_URI , slide_type, df_URI = None, local_save_dir = '/tmp'):
        """
        Summary: Convert tiles_fp and npy mask into tf records
        
        Arguments:
        df: DataFrame containing tiles_fp and npy mask fps
        output_URI: GCS URI or local path to save tf records
        df_URI (optional): GCS URI or local path to read csv
        
        """
        
        if df is not None:
            assert df_URI is None, ' Pick only df or df_URI as argument'
            self.df = df
        else:
            
            self.df_URI = df_URI
            self.df = pd.read_csv(self.df_URI,index_col=0)
            
        self.output_URI = output_URI
        self.df_train = self.df.query('train_valid == "train"').copy()
        self.df_train = self.df_train.sample(frac=1,random_state = 123).reset_index(drop=True) # shuffle df

        self.df_valid = self.df.query('train_valid == "valid"').copy()
        self.df_valid = self.df_valid.sample(frac=1,random_state = 123).reset_index(drop=True) # shuffle df
        
        self.slide_type = slide_type
        
    def df_to_tfrec(self,df, train_or_valid):

        tile_mask_fp_pairs = list(zip(df.tile_fp.values,df.mask_npy_fp.values))

        pairs_partitions = np.array_split(tile_mask_fp_pairs, np.ceil(len(tile_mask_fp_pairs)/5000))
        
        if not self.output_URI.startswith('gs'): # makedir if URI is not a GCS path
            os.makedirs(os.path.join(self.output_URI,train_or_valid),exist_ok=True)
            
        
        for i,partition in enumerate(pairs_partitions):
            # save tfrec in following naming: ae13_{stain_type}_{train or valid}_{number of partition}_{total number of pair of tile and mask in each tf record}}
            output_fp = os.path.join(self.output_URI,train_or_valid,f'ae13_{self.slide_type}_{train_or_valid}_{i}_{len(pairs_partitions)}.tfrec')
            writer = tf.io.TFRecordWriter(output_fp)

            for j,pair in enumerate(partition):
                tile_fp = pair[0]
                mask_npy_fp = pair[1]

                tile_bits = tf.io.read_file(tile_fp)
                mask_array = np.load(mask_npy_fp,allow_pickle=True).astype(np.uint8).ravel().tobytes()

                image = tf.image.decode_png(tile_bits,channels=3)
                image = tf.cast(image,tf.uint8)
                height = image.shape[0]
                width = image.shape[1]
                encoded_image = tf.image.encode_png(image)     

                example = tf.train.Example(features=tf.train.Features(feature={'image' :  _bytestring_feature([encoded_image.numpy()]), # .numpy() is needed to turn a tensor to byte
                                                                                "mask":        _bytestring_feature([mask_array]),              # one class in the list
                                                                                "size":          _int_feature([height, width])}))         # fixed length (2) list of ints



                writer.write(example.SerializeToString())

        writer.close()

    def run(self):
        self.df_to_tfrec(self.df_train,'train')
        self.df_to_tfrec(self.df_valid,'valid')