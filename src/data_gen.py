import sys
# scancer_homes = ['/home/jupyter/HER2/scancer', r'C:\Users\Curtis\Projects\HER2\scancer']
# for h in scancer_homes:
#     sys.path.insert(0,h)

from src.preprocessing.tile_generator import WSI
# this need to figure out

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
from tqdm import tqdm
import argparse
import logging
from functools import reduce

#logging.basicConfig(filename='log.log', filemode='w', level=logging.DEBUG)

class Tiler():
    def __init__(self):
        self.fp_slide = 'fp_slide'
        self.fp_mask = 'fp_mask'
        self.export_dir = 'export_dir'
        self.tile_size = 500
        self.downsample = 16
        self.DEBUG = False
        self.save_method = "cv" # "cv" , "plt" , "npy"
        
        # placeholder
        self.WSI = None
        self.base_image = None
        self.mask = None
        self.mask_r = None
        self.tile_addresses = None
    
    def set_config(self, config_di, show = True):
        for k,v in config_di.items():
            setattr(self, k,v)
        if show:
            self.show_config()
        
    def show_config(self):
        for k,v in self.__dict__.items():
            print(k,":", v)
            
    def load_slide(self):
        self.WSI = WSI(self.fp_slide)
        
    def load_mask(self):
        self.mask = cv.imread(self.fp_mask)
        
    def load_base_image(self):
        slide = self.WSI.slide
        levels = slide.level_downsamples
        level_index = self._get_best_level_for_downsample(levels, self.downsample)
        logging.debug(f"{level_index}, {self.downsample}")
        size_best = slide.level_dimensions[level_index]
        base_image = slide.get_thumbnail(size_best)
        base_image = np.array(base_image)
        self.base_image = base_image
        return base_image
    
    def export(self):
        # create out dirs
        tiles_dir = os.path.join(self.export_dir, "tiles","0")
        mask_dir = os.path.join(self.export_dir, "masks","0")
        npy_dir = os.path.join(self.export_dir, "npy","0")

        for mydir in [tiles_dir, mask_dir, npy_dir]:
            os.makedirs(mydir, exist_ok=True)

        # iterate tiles
        c=0
        for address in tqdm(self.tile_addresses):
            try:
                tile, tile_mask = self._create_tile(self.base_image, self.mask_r, *address)
                
                fp_tile = os.path.join(tiles_dir,f'{self.WSI.name}_{address}.png')
                fp_mask = os.path.join(mask_dir,f'{self.WSI.name}_{address}_mask.png')
                fp_npy = os.path.join(npy_dir,f'{self.WSI.name}_{address}_mask.npy')
                # plt changes pixel value when the mask is all 0 or all 1
                # cv.imwrite(fp_mask, tile_mask)
                # np.save(fp_mask.replace('png','npy'), tile_mask )
                
#                 if self.save_method == "npy":
#                     np.save(fp_tile.replace('png','npy'), tile)
#                     np.save(fp_mask.replace('png','npy'), tile_mask)
                if self.save_method == "plt":
                    plt.imsave(fp_tile, tile)
                    plt.imsave(fp_mask, tile_mask, vmin=0, vmax=1)
                    np.save(fp_npy, tile_mask)
                elif self.save_method == "cv":
                    cv.imwrite(fp_tile, tile)
                    cv.imwrite(fp_mask, tile_mask)
                    np.save(fp_npy, tile_mask)
                else:
                    break
                    print("specify a valid save method in self.save_method")


            except Exception as e:
                print(address, e)
            c+=1
            if self.DEBUG == True and c>=10:
                break
                print("DEBUG MODE ENABLED. LOOP STOPPED AT 10 TILES")

    
    def run(self):
        self.load_slide()
        self.load_mask()
        self.load_base_image()
        self.load_mask()
        self._match_img_and_mask()
        self._get_tile_addresses()
        self.export()

    ###  UNDER THE HOOD ###
    
    @staticmethod
    def _get_best_level_for_downsample(downsample, arr):
        """

        Finds the closest level of the image pyramid given a downsample factor

        tiler.WSI.slide.level_downsamples
        >>> (1.0, 4.0, 16.001696219633136, 32.00557123681207)

        my_downsample_factor = 4
        
        return:
        index = 1
        """
        L = np.array(arr)
        diff = np.absolute(L-downsample)
        index = np.argmin(diff)
        return index
    
    @staticmethod
    def _resize_imgs(img1, img2):
        """ for numpy arrays"""
        out = cv.resize(img1, img2.shape[:2][::-1])
        return out
    
    @staticmethod
    def _convert_rgb_to_binary_mask(mask):
        _, grey = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        return grey[:,:,0]
    
    def _match_img_and_mask(self):
        self.mask_r = self._resize_imgs(self.mask, self.base_image)
        
    def _create_tile(self, img, mask, x,y,w,h, mode="binary"):
        tile =       img[y:y+h, x:x+w, :]
        tile_mask = mask[y:y+h, x:x+w, :]
        if mode == "binary":
            tile_mask = self._convert_rgb_to_binary_mask(tile_mask)
            tile_mask = np.where(tile_mask>0,0,1)  # np.where(tile_mask>0,1,0) assign background as 1, tumor as 0
        return tile, tile_mask
    
    def _get_tile_addresses(self):
        tile_size = self.tile_size
        img = self.base_image
        img_H, img_W, _ = img.shape
        n_rows = img_H // tile_size
        n_cols = img_W // tile_size
        
        l_address = []
        for i in range(n_rows):
            for j in range(n_cols):
                address = j*tile_size, i*tile_size, tile_size, tile_size
                l_address.append(address)

        self.tile_addresses = l_address
        return l_address
    
class Tiler_trim(Tiler):
    
    def __init__(self):
        super().__init__()

    def load_json(self, min_A=1000):
        """
        qupath exported rectangles as polygon with 5 points. 
        simple fitler to get the actual rois, and remove small rectangular regions
        """

        with open(self.json) as f:
            data = geojson.load(f)
            
        df = pd.DataFrame(data)
        df['geo_type'] = df.geometry.apply(lambda x: x['type'])
        df['geo_shape'] = df.geometry.apply(lambda x: len(x['coordinates'][0]))
        df['shape'] = df.geometry.apply(self._get_area)
        df['class_'] = df.properties.apply(lambda x: x['classification']['name'])
        
        sql = f"geo_shape == 5 & shape >= {min_A}"
        df_roi = df.query(sql)        
        self.df_roi = df_roi
        return df_roi
    
    def get_tumor_stroma_shapes(self):
        df = self.df_roi
        self.tumor_shape = MultiPolygon([self._get_shape(i) for i in df[df.class_ == "Tumor"]['geometry']])
        self.stroma_shape = MultiPolygon([self._get_shape(i) for i in df[df.class_ == "Stroma"]['geometry']])

    def get_tumor_stroma_addresses(self):
        tumor_addresses = []
        stroma_addresses = []
        for a in self.tile_addresses:
            x,y,w,h = np.array(a) * self.downsample
            
            shape = box(x,y, x+w, y+h)
            if self.tumor_shape.contains(shape):
                tumor_addresses.append(a)
            elif self.stroma_shape.contains(shape):
                stroma_addresses.append(a)
            else:
                pass
        
        self.tumor_addresses = tumor_addresses
        self.stroma_addresses = stroma_addresses
        
    @staticmethod
    def _get_shape(x):
        xy = np.squeeze(x['coordinates'])
        shape = Polygon(xy)
        return shape
    
    @staticmethod
    def _get_area(roi):
        try:
            xy = roi['coordinates'][0]
            xy = np.squeeze(np.array(xy))
            box = Polygon(xy)
            return box.area

        # ValueError: A LinearRing must have at least 3 coordinate tuples
        # --> 434     assert (n == 2 or n == 3)
        #     435 
        #     436     # Add closing coordinates if not provided

        # AssertionError: 
        except (ValueError, AssertionError):
            return 0

        
    ### OVERRIDE ORIGINAL METHODS ###
    def _create_tile(self, img, mask, x,y,w,h, mode="binary", is_stroma=False):
        tile =       img[y:y+h, x:x+w, :]
        tile_mask = mask[y:y+h, x:x+w, :]
        if mode == "binary":
            tile_mask = self._convert_rgb_to_binary_mask(tile_mask)
            tile_mask = np.where(tile_mask>0,0,1)  # np.where(tile_mask>0,1,0) assign background as 1, tumor as 0
        if is_stroma:
            tile_mask = np.zeros(tile_mask.shape, "uint8")
        return tile, tile_mask
    
    def export(self, addresses, is_stroma=False):
 

        #[todo] addslide id in root dir
        # create out dirs
        tiles_dir = os.path.join(self.export_dir, "tiles")
        mask_dir = os.path.join(self.export_dir, "masks")
        npy_dir = os.path.join(self.export_dir, "npy")

        for mydir in [tiles_dir, mask_dir, npy_dir]:
            os.makedirs(mydir, exist_ok=True)

        # iterate tiles
        c=0
        for address in tqdm(addresses):
            try:
                tile, tile_mask = self._create_tile(self.base_image, self.mask_r, *address, is_stroma=is_stroma)
                
                fp_tile = os.path.join(tiles_dir,f'{self.WSI.name}_{address}.png')
                fp_mask = os.path.join(mask_dir,f'{self.WSI.name}_{address}_mask.png')
                fp_npy = os.path.join(npy_dir,f'{self.WSI.name}_{address}_mask.npy')
                # plt changes pixel value when the mask is all 0 or all 1
                # cv.imwrite(fp_mask, tile_mask)
                # np.save(fp_mask.replace('png','npy'), tile_mask )
                
#                 if self.save_method == "npy":
#                     np.save(fp_tile.replace('png','npy'), tile)
#                     np.save(fp_mask.replace('png','npy'), tile_mask)
                if self.save_method == "plt":
                    plt.imsave(fp_tile, tile)
                    plt.imsave(fp_mask, tile_mask, vmin=0, vmax=1)
                    np.save(fp_npy, tile_mask)
                elif self.save_method == "cv":
                    cv.imwrite(fp_tile, tile)
                    cv.imwrite(fp_mask, tile_mask)
                    np.save(fp_npy, tile_mask)
                else:
                    break
                    print("specify a valid save method in self.save_method")


            except Exception as e:
                print(address, e)
            c+=1
            if self.DEBUG == True and c>=10:
                break
                print("DEBUG MODE ENABLED. LOOP STOPPED AT 10 TILES")

    def export_tile_summary(self, csv_out, plot=False):
    
        df = pd.DataFrame([self.tile_addresses], columns=['address']).T
        df['slide_id'] = self.WSI.name
        df_t = pd.DataFrame([self.tumor_addresses]).T
        df_s = pd.DataFrame([self.stroma_addresses]).T 
        df_t['is_tumor'] = 1
        df_s['is_stroma'] = 1
        df_out = df.set_index(0).join(df_t.set_index(0)).join(df_s.set_index(0))
        df_out = df_out.reset_index()
        df_out.to_csv(csv_out)
        if plot:
            import seaborn as sns
            sns.heatmap(df_out.isna(),cmap="plasma")

            
    def run(self):
        self.load_slide()
        self.load_mask()
        self.load_base_image()
        self.load_mask()
        self.load_json()

        self._match_img_and_mask()
        self._get_tile_addresses()
        
        self.get_tumor_stroma_shapes()
        self.get_tumor_stroma_addresses()
        
        self.export(self.tumor_addresses)     
        self.export(self.stroma_addresses, is_stroma=True)        


class Tiler_ROI_DCIS(Tiler_trim):
    
    def __init__(self):
        super().__init__()
        self.DCIS_export_mode = "ignore" #ignore, as_stroma, as_dcis

        # check in qupath
        # https://www.colorhexa.com/ffc800
        self.cmap = {'stroma': (255, 255, 255),
                     'tumor': (200, 0, 0),
                      'dcis': (255, 200, 0)}
        self.class_index = {'stroma':0,
                             'tumor': 1,
                              'dcis': 2}
        self.mapper = self._create_mapper(self.cmap, self.class_index)
        self.disable_tqdm = False
    ########## UPDATED METHODS ##########
    def load_mask(self):
        mask = cv.imread(self.fp_mask)
        self.mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

    def get_tumor_stroma_shapes(self):
        """Disable old function names"""
        print("Deprecated. Use get_path_class_shapes() instead.")

    def get_tumor_stroma_addresses(self):
        """Disable old function names"""
        print("Deprecated. Use get_path_class_addresses() instead.")        

    def get_path_class_shapes(self):
        """
        create groups of rectangle shapes from ROI for each class
        """
        df = self.df_roi
        self.tumor_shape = MultiPolygon([self._get_shape(i) for i in df[df.class_ == "Tumor"]['geometry']])
        self.stroma_shape = MultiPolygon([self._get_shape(i) for i in df[df.class_ == "Stroma"]['geometry']])
        self.dcis_shape = MultiPolygon([self._get_shape(i) for i in df[df.class_ == "Other"]['geometry']])  

        # path_class_qupath_names = ["Tumor", "Stroma", "Other"]
        # for path_class in path_class_qupath_names:
        #     mpolygon = MultiPolygon([self._get_shape(i) for i in df[df.class_ == path_class]['geometry']])

        #     # replace name
        #     if path_class == "Other":
        #         path_class = "dcis"

        #     attr_name = path_class.lower() + "_shape"
        #     setattr(self, path_class, mpolygon)


    def get_path_class_addresses(self):
        """
        create groups of rectangle shapes from ROI for each class
        """
        tumor_addresses = []
        stroma_addresses = []
        dcis_addresses = []

        for a in self.tile_addresses:
            x,y,w,h = np.array(a) * self.downsample
            shape = box(x,y, x+w, y+h)
            
            if self.tumor_shape.contains(shape):
                tumor_addresses.append(a)
            elif self.stroma_shape.contains(shape):
                stroma_addresses.append(a)
            elif self.dcis_shape.contains(shape):
                dcis_addresses.append(a)
            else:
                pass
        
        self.tumor_addresses = tumor_addresses
        self.stroma_addresses = stroma_addresses
        self.dcis_addresses = dcis_addresses   
    
    def _create_tile(self, img, mask, x,y,w,h, mode="binary", is_stroma=False):
        """
        retrieve raw image data and raw mask, select location by address and chop into tiles
        """
        tile =       img[y:y+h, x:x+w, :]
        tile_mask = mask[y:y+h, x:x+w, :]

        if mode == "binary":
            tile_mask = self._convert_rgb_to_binary_mask(tile_mask)
            # np.where(tile_mask>0,1,0) assign background as 1, tumor as 0
            tile_mask = np.where(tile_mask>0,0,1)  

        elif mode == "multiclass":
            tile_mask = self._cvt_mask3d_to_mask2d(tile_mask, self.mapper, 0)
            tile_mask = tile_mask.astype('uint8')

        if is_stroma:
            tile_mask = np.zeros(tile_mask.shape, "uint8")

        return tile, tile_mask

    def export(self, addresses, is_stroma=False, mode="multiclass"):
        # added args for _export_tiles

        # create out dirs
        tiles_dir = os.path.join(self.export_dir, "tiles")
        mask_dir = os.path.join(self.export_dir, "masks")
        npy_dir = os.path.join(self.export_dir, "npy")

        for mydir in [tiles_dir, mask_dir, npy_dir]:
            os.makedirs(mydir, exist_ok=True)

        # iterate tiles
        c=0
        for address in tqdm(addresses,disable = self.disable_tqdm):
            try:
                tile, tile_mask = self._create_tile(self.base_image, self.mask_r, *address, is_stroma=is_stroma, mode=mode)
                
                fp_tile = os.path.join(tiles_dir,f'{self.WSI.name}_{address}.png')
                fp_mask = os.path.join(mask_dir,f'{self.WSI.name}_{address}_mask.png')
                fp_npy = os.path.join(npy_dir,f'{self.WSI.name}_{address}_mask.npy')
                # plt changes pixel value when the mask is all 0 or all 1
                # cv.imwrite(fp_mask, tile_mask)
                # np.save(fp_mask.replace('png','npy'), tile_mask )
                
#                 if self.save_method == "npy":
#                     np.save(fp_tile.replace('png','npy'), tile)
#                     np.save(fp_mask.replace('png','npy'), tile_mask)
                if self.save_method == "plt":
                    plt.imsave(fp_tile, tile)
                    plt.imsave(fp_mask, tile_mask, vmin=0, vmax=1)
                    np.save(fp_npy, tile_mask)
                elif self.save_method == "cv":
                    cv.imwrite(fp_tile, tile)
                    cv.imwrite(fp_mask, tile_mask)
                    np.save(fp_npy, tile_mask)
                else:
                    break
                    print("specify a valid save method in self.save_method")

            except Exception as e:
                print(address, e)
            c+=1
            if self.DEBUG == True and c >= 10:
                break
                print("DEBUG MODE ENABLED. LOOP STOPPED AT 10 TILES")

    def run(self):
        self.load_slide()
        self.load_mask()
        self.load_base_image()
        self.load_mask()
        self.load_json()

        self._match_img_and_mask()
        self._get_tile_addresses()
        
        self.get_path_class_shapes() 
        self.get_path_class_addresses() 
        
        self.export(self.tumor_addresses)     
        self.export(self.stroma_addresses, is_stroma=True)  

        # some options in handling dcis
        if self.DCIS_export_mode == "as_stroma":
            self.export(self.dcis_addresses, is_stroma=True)

        elif self.DCIS_export_mode == "as_dcis":
            self.export(self.dcis_addresses, is_stroma=False)

        elif self.DCIS_export_mode == "ignore":
            pass
        else:
            pass

    ##############################

    ########## NEW METHODS ##########
    @staticmethod
    def _cvt_mask3d_to_mask2d(m, mapper={(1,1,1):1, (2,2,2):2}, default = 0):
        """
        a = np.array(
                    [
                        [(1,1,1), (2,2,2), (0,0,0)],
                        [(1,1,1), (2,2,2), (0,0,0)],
                        [(1,1,1), (2,2,2), (0,0,0)],
                    ]
                )
        a
        o = cvt_mask3d_to_mask2d(a)
        plt.imshow(o)
        """
        #mapping
        sub_masks = []
        for k,v in mapper.items():
            # check elementwise equal
            sub_masks.append(np.where(np.all(m == k, axis=-1), v, default))

    #     o = reduce(np.add, sub_masks)
        o = reduce(np.bitwise_or, sub_masks)
        # o = o[:,:,0] # remove redundant channels
        return o


    @staticmethod
    def _create_mapper(cmap, class_index):
        mapper={}
        for k1, _ in zip(cmap, class_index):
            mapper.update({cmap[k1]:class_index[k1]})
        return mapper



    def visualize_mask_output(self, scale=10):

        mask = cv.imread(self.fp_mask)
        f = scale
        mask = mask[::f,::f,:]
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

        m_ = cvt_mask3d_to_mask2d(mask, mapper, 0)
        m_ = m_.astype('uint8')

        m_s = np.where(m_ == 0, 1, 0)
        m_t = np.where(m_ == 1, 1, 0)
        m_d = np.where(m_ == 2, 1, 0)

        # summary plot
        plt.imshow(m_, vmin=0, vmax=2)

        # breakdown plot
        fig, axes = plt.subplots(1,4)
        axes[0].imshow(m_, vmin=0, vmax=2)
        axes[1].imshow(m_s)
        axes[2].imshow(m_t)
        axes[3].imshow(m_d)

        axes[0].set_title("raw")
        axes[1].set_title("stroma")
        axes[2].set_title("tumor")
        axes[3].set_title("dcis")

        
class Tiler_NO_mask(Tiler):
    def __init__(self):
        super().__init__()
        
    def load_mask(self):
        pass
    
    def _create_tile(self, img, x,y,w,h , mask=None, mode="binary"):

        tile = img[y:y+h, x:x+w, :]
        return tile
    
    def export(self):
        # create out dirs
        tiles_dir = os.path.join(self.export_dir, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)

        # iterate tiles
        c=0
        for address in tqdm(self.tile_addresses):
            try:
                tile = self._create_tile(self.base_image, *address, mask=None)
                
                fp_tile = os.path.join(tiles_dir,f'{self.WSI.name}_{address}.png')

                if self.save_method == "plt":
                    plt.imsave(fp_tile, tile)
                elif self.save_method == "cv":
                    cv.imwrite(fp_tile, tile)
                else:
                    break
                    print("specify a valid save method in self.save_method")
                    
            except Exception as e:
                print(address, e)
            c+=1
            if self.DEBUG == True and c>=10:
                break
                print("DEBUG MODE ENABLED. LOOP STOPPED AT 10 TILES")
                
    def _match_img_and_mask(self):
        pass
    
    ##############################
    
    

def main():
    di = dict(export_dir=r'C:\Users\Angus\Desktop\AE13\tiles_mask',
        # fp_slide=r"F:\AE13\data\slides\06S01637 I-1.svs",
        # fp_mask=r"F:\AE13\notebooks\masks\06S01637 I-1-labels.png",
        fp_slide=r"C:\Users\Angus\Desktop\AE13\slides\06S18152 I-13 AE1,3.svs",
        fp_mask=r"C:\Users\Angus\Desktop\AE13\qpproj_10slides\export\06S18152 I-13 AE1,3-labels.png",
        tile_size=500,
        downsample=4,
        # DEBUG=True,
        save_method='plt',
        )

    tile_gen = Tiler()
    tile_gen.set_config(di)
    tile_gen.run()



if __name__ == "__main__":
    main()

    # parser = argparse.ArgumentParser(description="Tile generator for AE13")

    # parser.add_argument('fp_slide')
    # parser.add_argument('fp_mask')
    # parser.add_argument('--downsample_idx', default=2, type=int)
    # parser.add_argument('--downsample', default=16, type=int) # this is fake atm
    # parser.add_argument('--export_dir', default='train')
    # parser.add_argument('--tile_size', default=500, type=int)

    # args = parser.parse_args()
    # kwargs = args.__dict__
    # main(**kwargs)

