import tensorflow as tf
import os
import argparse
import pathos

import logging
from pathlib import Path

from mass_data_gen import main
from train_test_split import generate_train_valid_eval_csv
from df_to_tfrec import df_to_tfrec

def copy_data(slides_dir_URI, masks_dir_URI, json_dir_URI, slide_type = 'IHC', num_cpu = 4):

    jsons_URI = [js for js in tf.io.gfile.glob(f'{json_dir_URI}/*.json') if slide_type in js]
    valid_jsons_URI = [j for j in jsons_URI if tf.io.gfile.GFile(j).size() > 10]  # filter out 2B size jsons
    
    slide_ids = [Path(js).stem.split('.svs')[0] for js in valid_jsons_URI ] # get ids from IHC json
    
    masks_URI =  [mask for mask in tf.io.gfile.glob(f'{masks_dir_URI}/*.png') for slide_id in slide_ids if slide_id in mask] 
    
    slides_URI = [slide for slide in tf.io.gfile.glob(f'{slides_dir_URI}/*.svs') for slide_id in slide_ids if slide_id in slide]  # select only slide that have json

    local_slide_dir = '/tmp/slides/'
    local_masks_dir = '/tmp/masks/'
    local_jsons_dir = '/tmp/jsons/'
    os.makedirs(local_slide_dir,exist_ok=True)
    os.makedirs(local_masks_dir,exist_ok=True)
    os.makedirs(local_jsons_dir,exist_ok=True)
    
    def copy_slide_from_bucket(src):
        tf.io.gfile.copy(src,local_slide_dir,overwrite=True)

    def copy_mask_png_from_bucket(src):
        tf.io.gfile.copy(src,local_masks_dir,overwrite=True)

    def copy_json_from_bucket(src):
        tf.io.gfile.copy(src,local_jsons_dir,overwrite=True)
    

    p = pathos.pools.ProcessPool(num_cpu)
    p.map(copy_mask_png_from_bucket,masks_URI)
    p.map(copy_slide_from_bucket,slides_URI)
    p.map(copy_json_from_bucket, valid_jsons_URI)


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="AE13 tiling engine")

    parser.add_argument('--slides_URI', type = str, help = "GCS bucket URI of slides.")
    parser.add_argument('--json_URI', type = str,   help = "GCS bucket URI of jsons.")
    parser.add_argument('--mask_png_URI', type = str, help = "GCS bucket URI of masks.")
    parser.add_argument('--output_URI', type = str, help = "Output URI of bucket")
    parser.add_argument('--slide_type', type = str, help = " The stain of the slide, use to match json name with slide name if using a big directory for all jsons containing different type of stained slides")
    parser.add_argument('--dcis_export_mode', type = str, help = "How to handle dcis annotations, ignore, as_stroma or as_dcis")
    parser.add_argument('--num_cpu', type=int, default = 6, help= "Number of cpu for multiprocessing")
    parser.add_argument('--tile_size', type=int, default = 500, help= "Tile size for tile generation")
    parser.add_argument('--downsample', type=int, default = 4, help= "Downsample factor of WSI")
    
    args = parser.parse_args()
    assert args.dcis_export_mode in ['ignore', 'as_stroma', 'as_dcis'], "Only ignore, as_stroma or as_dcis is avaliable"
    num_cpu = int(args.num_cpu) if args.num_cpu < os.cpu_count() else os.cpu_count()
    
    logging.info('Copying files from bucket')
    copy_data(args.slides_URI, args.mask_png_URI,args.json_URI , slide_type = args.slide_type, num_cpu = num_cpu)

    
    slides_dir = '/tmp/slides/'
    masks_dir = '/tmp/masks'
    json_dir = '/tmp/jsons/'
    export_dir = '/tmp/tile_masks' # dir for exporting tile,mask,npy
    preview = False


    kwargs = dict(slides_dir = slides_dir,
                  masks_dir = masks_dir,
                  export_dir = export_dir,
                  json_dir = json_dir,
                  tile_size = args.tile_size,
                  downsample = args.downsample,
                  DCIS_export_mode = args.dcis_export_mode,
                  num_cpu = num_cpu,
                  preview=preview,
                  disable_tqdm = True
                 )
    # tile generation
    logging.info('Generating tiles')
    main(**kwargs)
    # train test split

    df = generate_train_valid_eval_csv(data_dir = export_dir, df_save_dir = args.output_URI, mode = 'slide',seed = 123)#,check_tile_mask_fp=True)
    
    logging.info('Converting to tf record')
    # covnert tiles into tf record
    tfrec_dir = '/tmp/tf_rec'
    df_to_tfrec(df,tfrec_dir,args.slide_type).run()
    # copy pipeline artifacts tiles,masks_png,npy,df,tfrec to bucket
    copy_tile_mask_cmd = f'gsutil -m cp -r {export_dir} {args.output_URI}'
    os.system(copy_tile_mask_cmd)
    
    copy_tfrec_cmd = f'gsutil -m cp -r {tfrec_dir} {args.output_URI}/tf_rec'
    os.system(copy_tfrec_cmd)
    
    tf.io.gfile.rmtree('/tmp')
