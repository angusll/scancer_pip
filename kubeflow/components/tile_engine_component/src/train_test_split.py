import sys
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

def check_dcis(fp):
    if 2 in np.unique(np.load(fp)):
        return True
    else:
        return False
    
def generate_train_valid_eval_csv(data_dir, df_save_dir, mode = 'tile', check_tile_mask_fp = False, seed = 123):
    mode = mode.lower()
    assert (mode == 'tile' or mode == 'slide'), 'Only tile or slide mode is available, check your argument'
    
#     if select_scn:
# #        tiles_dir = f'{data_dir}/{select_scn}' #d for d in glob.glob(f'/*') if d.endswith('tiles')][0]
#         list_of_tiles_fp = glob.glob(f'{data_dir}/tiles/*.png')        
#     else:
#        tiles_dir = [d for d in glob.glob(f'{data_dir}/*') if d.endswith('tiles')][0]
    list_of_tiles_fp = glob.glob(f'{data_dir}/*/tiles/*.png')

    df = pd.DataFrame(list_of_tiles_fp,columns=['fp'])
    df['scn'] = df['fp'].apply(lambda x: Path(x).stem.split('_')[0])
    df['address'] = df['fp'].apply(lambda x: Path(x).stem.split('_')[1])
    df['filename'] = df['fp'].apply(lambda x: Path(x).stem)

    df=df[['filename','scn','address','fp']].rename(columns={'fp':'tile_fp'})

    df['mask_fp'] = df['tile_fp'].apply(lambda x: x.replace('/tiles/','/masks/').split('.png')[0]+'_mask.png') # maskfp = {data_dir}/{scn_name}//mask/{filename}+_mask.png
    df['mask_npy_fp'] = df['tile_fp'].apply(lambda x: x.replace('/tiles/','/npy/').split('.png')[0]+'_mask.npy')
    
    # tile level train test split
    if mode == 'tile':
        # Train test split on df
        train_df,test_df = train_test_split(df,test_size = 0.2,random_state =seed)
        # assign a train / valid label 
        train_df['train_valid'] = 'train'
        test_df['train_valid'] = 'valid'
   
    # test level train test split
    elif mode == 'slide':
        train_n_valid,test = train_test_split(df.scn.unique(),test_size = 0.2,random_state =seed)
        train,valid = train_test_split(train_n_valid,test_size = 0.2,random_state =seed)
        
        train_df = df[df.scn.isin(train)].copy()
        train_df['train_valid'] = 'train'
        
        valid_df = df[df.scn.isin(valid)].copy()
        valid_df['train_valid'] = 'valid'
        
        test_df = df[df.scn.isin(test)].copy()
        test_df['train_valid'] = 'test'
        
        

    train_valid_test_df = pd.concat([train_df,valid_df, test_df])
    train_valid_test_df.sort_index(inplace=True)
    train_valid_test_df['with_dcis'] = train_valid_test_df.mask_npy_fp.map(check_dcis)
    
    if not df_save_dir.startswith('gs'): # makedir if URI is not a GCS path
        os.makedirs(df_save_dir,exist_ok=True)
        
    save_fp = os.path.join(df_save_dir,'train_valid_df.csv')
    train_valid_test_df.to_csv(save_fp)  
    
    if check_tile_mask_fp:
        log = []
        
        logging.info('Checking tiles')
        for tile_fp in tqdm(train_valid_test_df.tile_fp.values):
            im = Image.open(tile_fp)
            try:
                im.verify()
            except Exception:
                log.append(tile_fp) 
                
        logging.info('Checking mask')
        for tile_fp in tqdm(train_valid_test_df.mask_fp.values):
            im = Image.open(tile_fp)
            try:
                im.verify()
            except Exception:
                log.append(tile_fp) 
    
        if log:
            print(log)
        
    return train_valid_test_df

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="AE13 train test split")

    parser.add_argument('--data_dir',type = str, help= "Director of tiles")
    parser.add_argument('--df_save_dir',type = str, help= "Director of output csv")
    parser.add_argument('--mode', type = str, default='slide', help= "Mode of split, either tile level or slide level")
    parser.add_argument('--random_seed', type=int, default=123, help= "Seed of random train test split")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    df_save_dir = args.df_save_dir
    mode = args.mode
    seed = args.random_seed
    
    train_valid_df = generate_train_valid_eval_csv(data_dir,df_save_dir,check_tile_mask_fp = True, mode = mode, seed = seed)