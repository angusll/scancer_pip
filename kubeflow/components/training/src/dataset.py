import tensorflow as tf
from pathlib import Path
import numpy as np
import albumentations as A

class Unet_Dataset:
    
    def __init__(self,train_tile_fps,train_mask_fps, valid_tile_fps, valid_mask_fps,mask_type = 'png',batch_size = 32, input_size = (299,299,3),
                 original_image_shape = (500,500), multi_gpu = False, image_aug_type = 'Geometric + color',multi_class=False):
        
        self.train_tile_fps = train_tile_fps
        self.train_mask_fps = train_mask_fps
        assert len(self.train_tile_fps) == len(self.train_mask_fps), 'Len of train tiles and mask are not match, check the fps'
        self.valid_tile_fps = valid_tile_fps
        self.valid_mask_fps = valid_mask_fps
        assert len(self.valid_tile_fps) == len(self.valid_mask_fps), 'Len of valid tiles and mask are not match, check the fps'
        
        self.mask_type = mask_type
        assert self.mask_type in ['png','npy'], 'The only type for mask are png or npy'
        self.multi_class = multi_class
        
        
        for list_tile_fps in [self.train_tile_fps,self.valid_tile_fps]:
            for fp in list_tile_fps:
                assert Path(fp).suffix == '.png', 'There are non-png file in fps, it will break dataset pipeline, check your input'
                
        for list_mask_fps in [self.train_mask_fps, self.valid_mask_fps]:
            for fp in list_mask_fps:              
                assert Path(fp).suffix == f'.{mask_type}', f'There are non-{mask_type} file in fps OR mask_fps does not match mask_type, this will break dataset pipeline, check your input'
        
        self.image_aug_type = image_aug_type
        assert self.image_aug_type in ['None','Geometric','Geometric + color'], 'Only None, Geometric or Geometric + color is available for image_aug_type, check your input'
        
        self.multi_gpu = multi_gpu
        if self.multi_gpu:
            self.mirrored_strategy = tf.distribute.MirroredStrategy()
            self.mirrored_strategy.num_replicas_in_sync
            self.BATCH_SIZE_PER_REPLICA = batch_size
            self.global_batch_size = (self.BATCH_SIZE_PER_REPLICA * self.mirrored_strategy.num_replicas_in_sync)
        else:
            self.global_batch_size = batch_size
            
        self.input_size = input_size
        self.original_image_shape = original_image_shape
        self.N_train = len(self.train_tile_fps) # total N of training
        self.N_valid = len(self.valid_tile_fps) # total N of training
        
        self.train_steps = tf.math.ceil(self.N_train / self.global_batch_size)
        self.valid_steps = tf.math.ceil(self.N_valid / self.global_batch_size)
        
        self.AUTO = tf.data.AUTOTUNE
        
    def read_images_mask(self,tile_fp, mask_fp):
        # read and decode tiles and mask from disk
        img = tf.io.read_file(tile_fp)
        img = tf.io.decode_png(img,channels = 3)
        
        # decode png mask to binary mask
        if self.mask_type == 'png':
            mask = tf.io.read_file(mask_fp)
            mask = tf.io.decode_jpeg(mask,channels = 1)
            mask = tf.where(mask < 200, tf.cast(0,tf.uint8), mask)  # 0 need to convert to tf.float32 for dtype compatbility
            mask = tf.where(mask >= 200, tf.cast(1,tf.uint8), mask) # 1 need to convert to tf.float32 for dtype compatbility
            mask = tf.image.resize(mask,[self.input_size[0],self.input_size[1]],preserve_aspect_ratio=True)
        
        # decode npy mask to binary mask
        elif self.mask_type == 'npy':
            mask = tf.py_function(lambda x: np.load(x.numpy().decode()),[mask_fp],tf.uint8) # mask is in numpy format       
            mask = tf.expand_dims(mask,-1) # this expand an extra dimension to make the shape of mask = [512,512,1] for resizing
            mask.set_shape([self.original_image_shape[0],self.original_image_shape[1],1])
        
        return img, mask
    
    def one_hot_multiclass_mask(self,image,mask):
        mask = tf.numpy_function(func=tf.keras.utils.to_categorical, inp=[mask,3], Tout=tf.float32,name = 'one_hot_mask')
        mask.set_shape([self.original_image_shape[0],self.original_image_shape[1],3])
        return image,mask
    

    def decode_normalise_images(self,tile_fp):
        # for pred dataset
        img = tf.io.read_file(tile_fp)
        img = tf.io.decode_png(img,channels = 3)
        img = tf.image.resize(img,[self.input_size[0],self.input_size[1]],preserve_aspect_ratio=True)
        img /= 255
        return img

    
    #################### Augmentation functions #################################
    def geometric_augmentation_function(self,img,mask):    
        
        aug = A.Compose([
                        A.transforms.Flip(always_apply=True),
                        A.RandomRotate90(always_apply=True),
                        A.Resize(self.original_image_shape[0],self.original_image_shape[1],always_apply=True)])

        augmented_img_mask = aug(image=img, mask=mask) # take 2d binary mask only eg. [500,500] not [500,500,1]

        img = augmented_img_mask['image']
        img = img[:,:,:3] # img has 4 channels after augmentation 
        
        mask = augmented_img_mask['mask']   
        mask = tf.cast(mask,tf.uint8) # cast mask to uint8 to ensure its still a binary mask
                       
        return img, mask#, img.shape, mask.shape 

    def geometric_image_aug(self,img,mask):
        aug_img,aug_mask = tf.numpy_function(func=self.geometric_augmentation_function, inp=[img, mask], Tout=[tf.uint8,tf.uint8],name = 'Geometric_augmentation')
        aug_img.set_shape([self.original_image_shape[0],self.original_image_shape[1],3])
        aug_mask.set_shape([self.original_image_shape[0],self.original_image_shape[1],1])
        
        return aug_img,aug_mask
        
    def colour_augmentation_function(self,img,mask):

        aug = A.Compose([
                A.augmentations.transforms.ColorJitter(brightness=(0.8,1),
                                                        contrast=(0.8,1.1),
                                                        saturation=0,
                                                        hue=0,
                                                        always_apply=False,
                                                        p = 0.5)])

        augmented_img = aug(image=img)
        img = augmented_img['image']   
        img = img[:,:,:3] # img has 4 channels after augmentation
        return img, mask
    
    def colour_image_aug(self,img,mask):
        
        aug_img,aug_mask= tf.numpy_function(func=self.colour_augmentation_function, inp=[img, mask], Tout=[tf.uint8,tf.uint8],name = 'Colour_augmentation')
        aug_img.set_shape([self.original_image_shape[0],self.original_image_shape[1],3])
        aug_mask.set_shape([self.original_image_shape[0],self.original_image_shape[1],1])
        
        return aug_img,aug_mask
    
                                                                                       
    def resize_normalise_img_mask(self,image,mask):        
        
        image = tf.image.resize(image,[self.input_size[0],self.input_size[1]])
        
#        mask = tf.expand_dims(mask,-1) # this expand an extra dimension to make the shape of mask = [512,512,1] for resizing
        mask = tf.image.resize(mask,[self.input_size[0],self.input_size[1]],preserve_aspect_ratio=True)
        image /= 255
        mask = tf.cast(mask,tf.float32)
        return image,mask
    
   #################################################   
   ###             Prepare Dataset               ###                                          
   #################################################

    def prepare_dataset(self):
        #### train ds #####
        train_tiles_fp_ds = tf.data.Dataset.from_tensor_slices(self.train_tile_fps)
        train_mask_fp_ds = tf.data.Dataset.from_tensor_slices(self.train_mask_fps)
        train_tiles_mask_fps_ds = tf.data.Dataset.zip((train_tiles_fp_ds,train_mask_fp_ds))
        
        train_ds = train_tiles_mask_fps_ds.shuffle(self.N_train).prefetch(self.AUTO).repeat() # shuffle and prefetch before mapping for memory efficiency 
        train_ds = train_ds.map(self.read_images_mask,num_parallel_calls=self.AUTO,deterministic=False)
        
        if self.image_aug_type == "Geometric":
            train_ds = train_ds.map(self.geometric_image_aug,num_parallel_calls=self.AUTO)
            
        elif self.image_aug_type == "Geometric + color":
            train_ds = train_ds.map(self.geometric_image_aug,num_parallel_calls=self.AUTO)
            train_ds = train_ds.map(self.colour_image_aug,num_parallel_calls=self.AUTO)
        
        if self.multi_class:
            train_ds = train_ds.map(self.one_hot_multiclass_mask,num_parallel_calls=self.AUTO)  
            
        train_ds = train_ds.map(self.resize_normalise_img_mask,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = train_ds.batch(self.global_batch_size)
        
        #### valid ds #####
        valid_tiles_fp_ds = tf.data.Dataset.from_tensor_slices(self.valid_tile_fps)
        valid_mask_fp_ds = tf.data.Dataset.from_tensor_slices(self.valid_mask_fps)
        valid_tiles_mask_fps_ds = tf.data.Dataset.zip((valid_tiles_fp_ds,valid_mask_fp_ds))
        
        valid_ds = valid_tiles_mask_fps_ds.shuffle(self.N_valid).prefetch(tf.data.experimental.AUTOTUNE).repeat() # shuffle and prefetch before mapping for memory efficiency 
        valid_ds = valid_ds.map(self.read_images_mask,num_parallel_calls=self.AUTO,deterministic=False)
        
        if self.multi_class:
            valid_ds = valid_ds.map(self.one_hot_multiclass_mask,num_parallel_calls=self.AUTO)      
        
        valid_ds = valid_ds.map(self.resize_normalise_img_mask,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        valid_ds = valid_ds.batch(self.global_batch_size)
        
        return train_ds, valid_ds
       
    def make_pred_dataset(self, tile_fps):
        tile_fps_ds = tf.data.Dataset.from_tensor_slices(tile_fps)
        ds = tile_fps_ds.prefetch(tf.data.experimental.AUTOTUNE) # shuffle and prefetch before mapping for memory efficiency 
        ds = ds.map(self.decode_normalise_images,num_parallel_calls=tf.data.experimental.AUTOTUNE,deterministic=True).batch(self.global_batch_size)
        return ds

    
class Unet_TFrecord_Dataset(Unet_Dataset):
    
    def __init__(self,train_tfrec_URI,valid_tfrec_URI,batch_size = 32, input_size = (299,299,3), 
                 original_image_shape = (500,500) ,multi_gpu = False,image_aug_type = 'Geometric'):
        
        self.train_tfrec_URI = train_tfrec_URI
        self.valid_tfrec_URI = valid_tfrec_URI
        self.batch_size = batch_size 
        self.input_size = input_size
        self.original_image_shape = original_image_shape
        self.multi_gpu = multi_gpu
        self.image_aug_type = image_aug_type
        assert self.image_aug_type in ['None','Geometric','Geometric + color'], 'Only None, Geometric or Geometric + color is available for image_aug_type, check your input'
        self.AUTO = tf.data.AUTOTUNE
        
        
        self.train_tfrec_filenames = tf.io.gfile.glob(self.train_tfrec_URI + '/*.tfrec')
        self.valid_tfrec_filenames = tf.io.gfile.glob(self.valid_tfrec_URI + '/*.tfrec')
        # add check empty filenames
        
        self.n_train = sum([int(Path(filename).stem.rsplit('_')[-1]) for filename in self.train_tfrec_filenames])
        self.n_valid = sum([int(Path(filename).stem.rsplit('_')[-1]) for filename in self.valid_tfrec_filenames])
        
        self.train_steps = np.ceil(self.n_train / self.batch_size)
        self.valid_steps = np.ceil(self.n_valid / self.batch_size)
        
    def decode_tfrecord(self,example):
        features = {
                "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
                "size":  tf.io.FixedLenFeature([2], tf.int64),  # two integers
                "mask": tf.io.FixedLenFeature([],tf.string)        # a certain number of floats
            }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        # FixedLenFeature fields are now ready to use: exmple['size']
        # VarLenFeature fields require additional sparse_to_dense decoding

        image = tf.image.decode_png(example['image'], channels=3)
        height = example['size'][0]
        width  = example['size'][1]
        mask = tf.reshape(tf.io.decode_raw(example['mask'],tf.uint8),[height,width])
                
        return image,mask 
    

    
    def prepare_dataset(self):
        # read from TFRecords. For optimal performance, read from multiple
        # TFRecord files at once and set the option experimental_deterministic = False
        # to allow order-altering optimizations.

        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False

        train_tfrec_ds = tf.data.TFRecordDataset(self.train_tfrec_filenames, num_parallel_reads=self.AUTO)
        train_ds = train_tfrec_ds.with_options(option_no_order)
        train_ds = train_ds.map(self.decode_tfrecord, num_parallel_calls=self.AUTO).repeat()
        
        if self.image_aug_type == "Geometric":
            train_ds = train_ds.map(self.geometric_image_aug,num_parallel_calls=self.AUTO)
            
        elif  self.image_aug_type == "Geometric + color":
            train_ds = train_ds.map(self.geometric_image_aug,num_parallel_calls=self.AUTO)
            train_ds = train_ds.map(self.colour_image_aug,num_parallel_calls=self.AUTO)

        train_ds = train_ds.map(self.resize_normalise_img_mask,num_parallel_calls=self.AUTO)
        train_ds = train_ds.shuffle(self.train_steps).batch(self.batch_size).prefetch(self.AUTO)
        
        valid_tfrec_ds = tf.data.TFRecordDataset(self.valid_tfrec_filenames, num_parallel_reads=self.AUTO)
        valid_ds = valid_tfrec_ds.with_options(option_no_order)
        valid_ds = valid_ds.map(self.decode_tfrecord, num_parallel_calls=self.AUTO).repeat()
        valid_ds = valid_ds.map(self.resize_normalise_img_mask,num_parallel_calls=self.AUTO)
        valid_ds = valid_ds.shuffle(self.valid_steps).batch(self.batch_size).prefetch(self.AUTO)
        
        return train_ds,valid_ds