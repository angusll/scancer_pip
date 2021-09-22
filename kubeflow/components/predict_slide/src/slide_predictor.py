import sys
import geojson
import json
import glob
from shapely.geometry import Polygon, MultiPolygon, box
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import shapely
from collections import Counter
import os
import argparse
import skimage.io
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tensorflow as tf
import logging
import argparse
from pathlib import Path
import pathos
from color import Color_Adjsutor
from data_gen import Tiler

#from unet.segmentation_models import build_unet_model

###### utils for kubeflow #########
def copy_slide_from_bucket(src,dst):
        tf.io.gfile.copy(src,dst,overwrite=True)
    

class Predict_tiler(Tiler):
    def __init__(self):
        super().__init__()
    
    def load(self):
        self.load_slide()
        self.load_base_image()
        self._get_tile_addresses()

    def gen_tiles_to_predict(self):
        for address in self.tile_addresses:
            tile = self._create_tile(self.base_image, *address)
            yield tile
  
    def _create_tile(self, img, x,y,w,h, mode="binary"):
        tile = img[y:y+h, x:x+w, :]
        return tile
    
class slide_predictor:
       
    def __init__(self, model_weights_fp:str,threshold:float,save_dir:str, colour_adjust: bool = False, tile_size:int = 500, downsample:int = 4):
        self.model_weights_fp = model_weights_fp
        self.threshold = threshold
        self.save_dir = save_dir
        self.colour_adjust = colour_adjust
        self.tile_size = 500
        self.downsample = 4
        self.model = tf.keras.models.load_model(self.model_weights_fp, compile=False)#build_unet_model(weights = config.model_weights_fp)
        self.model_input_size =  self.model.input_shape[1:3]
        
    def normalise_resize_image(self,image_array):
        img = tf.image.resize(image_array,[self.model_input_size[0],self.model_input_size[1]])
        img = tf.cast(img/255,tf.float32)
        return img

    def stitch_masks(self,y_hat_npys,addresses):

        def clean_pred(x):
            x = np.squeeze(x)
            x= cv.resize(x, (self.tile_size,self.tile_size))
            return x

        preds_arrays_clean = [clean_pred(y_hat) for y_hat in y_hat_npys]

        x_max = max(np.array(addresses)[:,0])
        y_max = max(np.array(addresses)[:,1])
        w_max = max(np.array(addresses)[:,2])
        h_max = max(np.array(addresses)[:,3])

        W = x_max + w_max
        H = y_max + h_max

        raw_mask = np.zeros((H, W), dtype='float32')
        for address, array in zip(addresses, preds_arrays_clean):
            x,y,w,h = address
            # simple value assignemnt in numpy
            raw_mask[y:y+h, x:x+w] = array
        return raw_mask

    def gen_multiply_binary_mask(self,raw_mask,threshold):
        binary_mask = np.where(raw_mask>threshold, 1,0)
        return binary_mask

            

    def predict(self,slide_fp):
        """
        Predict a slide and save the probability mask and 0.1 - 0.9 binary mask locally
        """
        def color_adj_wrapper(image):
            image = tf.py_function(func=adjustor.adjust, inp=[image], Tout=tf.float32)
            image.set_shape([self.tile_size,self.tile_size,3])
            return image
        


        tiler_config = dict(fp_slide=slide_fp,
                            tile_size=self.tile_size,
                            downsample=self.downsample)

        self.tiler = Predict_tiler()
        self.tiler.set_config(tiler_config)
        self.tiler.load()
        
            
        BATCH_SIZE = 32
        ds = tf.data.Dataset.from_generator(self.tiler.gen_tiles_to_predict,output_types=tf.uint8,output_shapes=[self.tile_size,self.tile_size,3])
        
        if self.colour_adjust:
            ref_tile = skimage.io.imread('/ae13/predictor/ref_tile.png')[:,:,:3]
            adjustor = Color_Adjsutor(ref_tile)
            ds = ds.map(color_adj_wrapper,num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        ds = ds.map(self.normalise_resize_image,num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        y_hat = self.model.predict(ds,verbose=2)
        
        prob_mask_local_fp, list_of_local_binary_mask_fp = self.save_mask(y_hat)
        
        tf.keras.backend.clear_session()
        del y_hat
        
        return prob_mask_local_fp, list_of_local_binary_mask_fp
        
        ############ Generate mask and save ###################
    def save_mask(self,y_hat):
        
        local_prob_dir = os.path.join(self.save_dir,'prob_masks')
        local_binary_dir = os.path.join(self.save_dir,'binary_masks')
        os.makedirs(local_prob_dir,exist_ok=True)
        os.makedirs(local_binary_dir,exist_ok=True)

        prob_mask = self.stitch_masks(y_hat,self.tiler.tile_addresses)#,threshold = config.threshold)
        prob_mask_local_fp = f'{local_prob_dir}/{self.tiler.WSI.name}_Tumor_({self.downsample},0,0,0,0)-prob-mask.png'
        plt.imsave(prob_mask_local_fp,prob_mask)

        if len(prob_mask.shape) == 2: # two class [0,1], binary mask
            list_of_local_binary_mask_fp = []

            for thres in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

                binary_mask_local_fp = f'{local_binary_dir}/{self.tiler.WSI.name}_{thres}_Tumor_({self.downsample},0,0,0,0)-mask.png'

                binary_mask = self.gen_multiply_binary_mask(raw_mask = prob_mask,threshold = thres)

                cv.imwrite(binary_mask_local_fp,binary_mask*255) # binary mask * 255 to save it as black white mask
                list_of_local_binary_mask_fp.append(binary_mask_local_fp)

        else:
            class_index = {0:'Stroma',1:'Tumor',2:'Others'}
            for i in range(pred_one_hot.shape[-1]):
                prob_mask_local_fp = f'{save_dir_id}/prob_mask/{self.tiler.WSI.name}_{class_index[i]}_({self.downsample},0,0,0,0)-prob-mask.png'
                binary_mask_local_fp = f'{save_dir_id}/binary_mask/{self.tiler.WSI.name}_{class_index[i]}_({self.downsample},0,0,0,0)-mask.png'
                binary_mask = self.gen_multiply_binary_mask()
                plt.imsave(prob_mask_local_fp,prob_mask[:,:,i])
                cv.imwrite(binary_mask_local_fp,binary_mask[:,:,i]*255) # binary mask * 255 to save it as black white mask

        return prob_mask_local_fp, list_of_local_binary_mask_fp
    
def batch_predict(list_of_slide_uris,local_slide_dir,local_save_dir, output_dir_uri):
    """
    Summary: Predict on a list of slide uris. Each uri must be a direct uri of the slide object.
             Save the predicted binary and probability mask in save dir.
    """

    for slide_uri in list_of_slide_uris:
        
        slide_id = Path(slide_uri).name
                
        logging.info(f'Working on {slide_id}')
        
        local_slide_fp = os.path.join(local_slide_dir,slide_id)
        copy_slide_from_bucket(slide_uri,local_slide_fp)
        
        prob_mask_local_fp, list_of_local_binary_mask_fp = predict(local_slide_fp,local_save_dir)
        
        # copy every thing predicted to bucket
        prob_mask_basename = Path(prob_mask_local_fp).name
        prob_mask_output_fp_uri = os.path.join(output_dir_uri,'prob_mask',prob_mask_basename)
        logging.info(f'Saved at  {prob_mask_output_fp_uri}')
        tf.io.gfile.copy(prob_mask_local_fp,prob_mask_output_fp_uri,overwrite=True)
            
        for mask_fp in list_of_local_binary_mask_fp:
            mask_basename = Path(mask_fp).name
            output_fp_uri = os.path.join(output_dir_uri,'binary_mask',mask_basename)
            logging.info(f'Saved at  {output_fp_uri}')
            tf.io.gfile.copy(mask_fp,output_fp_uri,overwrite=True)        
        


        
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="AE13 tiling engine")
    
    parser.add_argument('--experiment_id', type = str, help= "Unique id of experiment")
    parser.add_argument('--slides_URI', type = str, help = "GCS bucket URI of slides.")
    parser.add_argument('--slide_type', type = str, help = "IHC or HE slides")
    parser.add_argument('--train_valid_csv_URI', type = str, help= "GCS bucket URI of train valid csv. Just to get the scn id for prediction. Enter 'all' for predict the whole bucket")
    parser.add_argument('--num_of_slides' , type = int, default = 20, help = 'Number of slides to predict if uses whole bucket. -1 for all slides. No effect if provide csv URI')
    parser.add_argument('--output_URI', type = str, help = "Output URI of bucket. Prediction masks will save at {output_URI}/results/{experiment_id}")
    parser.add_argument('--model_weights_URI', type = str, help = "Model weight .h5 URI")
    parser.add_argument('--threshold', type = float, help = "Threshold for binarising probability mask")
    parser.add_argument('--colour_adjust', type = bool, help = "To use hed histogram matching or not", default = False)
    
    args = parser.parse_args()
    
    # define local dirs

    local_dir  = "/tmp"

    local_scn_dir = f'{local_dir}/eval_scn'
    os.makedirs(local_scn_dir,exist_ok=True)
    
    local_ckpt_dir = f'{local_dir}/model_checkpoint'
    os.makedirs(local_ckpt_dir,exist_ok=True)
    
    local_predict_save_dir = os.path.join(local_dir,'prediction_results')
    os.makedirs(local_predict_save_dir,exist_ok=True)
    
    
    if args.train_valid_csv_URI.lower() == 'all':
        if args.num_of_slides == -1:
            scns_to_predict = tf.io.gfile.glob(os.path.join(args.slides_URI,'*svs'))
        else:
            scns_to_predict = tf.io.gfile.glob(os.path.join(args.slides_URI,'*svs'))[:args.num_of_slides]
            
        logging.info(f'Predicting {len(scns_to_predict)} of slides')
    else:
        df = pd.read_csv(args.train_valid_csv_URI)
        scns_to_predict = [os.path.join(args.slides_URI,scn) for scn in df.query('train_valid == "test"').scn.unique()]
    
    # copy model ckpt from bucket
    local_model_ckpt_fp = f'{local_ckpt_dir}/{Path(args.model_weights_URI).name}'
    tf.io.gfile.copy(args.model_weights_URI,local_model_ckpt_fp,overwrite=True)
    

    predictor = slide_predictor(
                                model_weights_fp = local_model_ckpt_fp,
                                threshold = args.threshold,
                                save_dir = local_predict_save_dir,
                                colour_adjust = args.colour_adjust)
        
    for slide_uri in scns_to_predict:
        
        slide_id = Path(slide_uri).name
                
        logging.info(f'Working on {slide_id}')
        
        local_scn_fp = os.path.join(local_scn_dir,slide_id)
        copy_slide_from_bucket(slide_uri,local_scn_fp)
        
        prob_mask_local_fp, list_of_local_binary_mask_fp = predictor.predict(local_scn_fp)
        
        prob_mask_basename = Path(prob_mask_local_fp).name
        prob_mask_output_fp_uri = os.path.join(args.output_URI,args.experiment_id,'prob_mask',prob_mask_basename)
        logging.info(f'Saved at  {prob_mask_output_fp_uri}')
        tf.io.gfile.copy(prob_mask_local_fp,prob_mask_output_fp_uri,overwrite=True)
            
        for mask_fp in list_of_local_binary_mask_fp:
            mask_basename = Path(mask_fp).name
            output_fp_uri = os.path.join(args.output_URI,'slide_predictions',args.experiment_id,'binary_mask',mask_basename)
            logging.info(f'Saved at  {output_fp_uri}')
            tf.io.gfile.copy(mask_fp,output_fp_uri,overwrite=True)        
    
    
    tf.io.gfile.rmtree('/tmp')