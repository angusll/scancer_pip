import tensorflow as tf
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import logging
import argparse
import json


def IoU_Score(y_true,y_pred):
    
    y_true = tf.cast(y_true,tf.int32) # must be int32 or float32 or above, uint8 simply cannot have long length of integer, it will clip to int8 length
    y_pred = tf.cast(y_pred,tf.int32) 
    
    # check size of y true mask and pred mask, pred mask might be smaller as prediction process trimmed out non square tiles
    if y_true.shape[:2] != y_pred.shape[:2]:
        y_true = y_true[:y_pred.shape[0],:y_pred.shape[1]] # crop right and bottom boundaries of y_true to match the size of y pred
    
    # Flatten    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    intersection = tf.cast(intersection,tf.float32)
    union = tf.cast(union,tf.float32)
    
    iou = intersection/(union+tf.keras.backend.epsilon())
    return tf.cast(iou,tf.float32).numpy()


def evaluate(tile_mask_uri,train_valid_csv_uri, model_uri, threshold, output_uri, experiment_id,slide_type):
    
    def decode_normalise_images(tile_fp):
        img = tf.io.read_file(tile_fp)
        img = tf.io.decode_png(img,channels = 3)
        img = tf.image.resize(img,[model.input_shape[1],model.input_shape[2]],preserve_aspect_ratio=True)
        img /= 255
        return img
    
    local_dir  = "/tmp"

    local_tile_mask_dir = f'{local_dir}/eval_scn'
    os.makedirs(local_tile_mask_dir,exist_ok=True)
    
    local_ckpt_dir = f'{local_dir}/model_checkpoint/'
    os.makedirs(local_ckpt_dir,exist_ok=True)
    
    df = pd.read_csv(train_valid_csv_uri)
    eval_scns = [Path(scn).stem for scn in df.query('train_valid == "test"').scn.unique()]
    
    # copy tiles and npys from bucket
    for eval_scn in eval_scns:
        copy_cmd = f'gsutil -m cp -r "{tile_mask_uri}/{eval_scn}/" {local_tile_mask_dir}'
        os.system(copy_cmd)
    
    model_ckpt_URI = tf.io.gfile.glob(os.path.join(output_uri,f'checkpoint/{experiment_id}/*.h5'))[0]

    # copy model ckpt from bucket
    tf.io.gfile.copy(model_ckpt_URI,f'{local_ckpt_dir}/{Path(model_ckpt_URI).name}',overwrite=True)

    model = tf.keras.models.load_model(f'{local_ckpt_dir}/{Path(model_ckpt_URI).name}',compile=False)

    score_df_list = []
    for scn in eval_scns:
        
        tile_fps = tf.io.gfile.glob(f"{local_tile_mask_dir}/{scn}/*tiles/*png")
        npy_fps = [fp.replace('tiles','npy').replace('.png','_mask.npy') for fp in tile_fps]
        tile_size = skimage.io.imread(tile_fps[0]).shape
        
        batch_size = 32
        pred_ds = tf.data.Dataset.from_tensor_slices(tile_fps).map(decode_normalise_images,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

        y_hat = model.predict(pred_ds,verbose=2)

        pred_masks = np.array([skimage.transform.resize(pred,(tile_size[0],tile_size[1])) for pred in y_hat])

        pred_binary_masks = np.where(pred_masks > threshold, 1, 0)

        zipped_npy_fp_y_hat = list(zip(npy_fps,pred_binary_masks))

        iou_with_tumor = []
        iou_stroma = []
        for fp, pred_mask in zipped_npy_fp_y_hat:
            y_true = np.load(fp)

            if (y_true.sum() == 0):
                score = IoU_Score(y_true,pred_mask)
                iou_stroma.append(score)

            else:
                score = IoU_Score(y_true,pred_mask)
                iou_with_tumor.append(score)

        iou_stroma = np.array(iou_stroma)
        iou_with_tumor = np.array(iou_with_tumor)

        if (iou_stroma.sum() == 0):
            iou_stroma = 1  # iou stroma are likely =0 as the intersection of 0 and 0 = 0 
        else:
            iou_stroma = iou_stroma.mean()

        mean_iou_dict  = {'iou_stroma':iou_stroma,
                         'iou_with_tumor':iou_with_tumor.mean()}

        score_df = pd.DataFrame({scn:mean_iou_dict}).T
        score_df_list.append(score_df )

    slides_score_df = pd.concat(score_df_list)
    slides_score_df['mean_iou'] = (slides_score_df.iou_stroma+slides_score_df.iou_with_tumor)/2
    slides_score_df.loc['overall_mean'] = slides_score_df.mean()
    
    score_df_save_uri = os.path.join(output_uri,f'evaluation/{slide_type}/id_{experiment_id}/iou.csv')
    slides_score_df.to_csv(score_df_save_uri)
    
    return score_df_save_uri
                                     
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="AE13 evaluation pipeline")
    # Input Output paths
    parser.add_argument('--tile_mask_uri', type = str, help= "GCS bucket URI of tile and masks.")
    parser.add_argument('--train_valid_csv_uri', type = str, help= "GCS bucket URI of train valid csv.")
    parser.add_argument('--output_uri', type = str, help = "Output URI of bucket")
    parser.add_argument('--model_uri', type = str, help = "GCS URI of model")
    parser.add_argument('--experiment_id', type = str, help = "Unique id of experiment")
    parser.add_argument('--slide_type', type = str, help = "IHC or HE slides")
    parser.add_argument('--threshold', type=float, default = 0.5, help= "Threshold for predicted mask")

    args = parser.parse_args()

    score_df_save_uri = evaluate(tile_mask_uri = args.tile_mask_uri,
                                train_valid_csv_uri = args.train_valid_csv_uri, 
                                 model_uri = args.model_uri,
                                 threshold = args.threshold,
                                 experiment_id = args.experiment_id, 
                                 slide_type = args.slide_type,
                                 output_uri = args.output_uri)
    
    metadata = {
                'outputs' : [{
                'type': 'table',
                'storage': 'gcs',
                'format': 'csv',
                'header': ['Unnamed: 0','iou_stroma','iou_with_tumor'],
                'source': score_df_save_uri
                }]
              }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)