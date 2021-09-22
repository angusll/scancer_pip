import sys
import glob
import os
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import tensorflow as tf

from datetime import datetime
import albumentations as A

from dataset import Unet_TFrecord_Dataset

#from unet.model import get_unet_model
from models import build_unet_model

import segmentation_models as sm
sm.set_framework('tf.keras')



if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="AE13 training pipeline")
    # Input Output paths
    parser.add_argument('--train_tfrec_uri', type = str, help= "GCS bucket URI of train tfrecords.")
    parser.add_argument('--valid_tfrec_uri', type = str,   help= "GCS bucket URI of valid tfrecords.")
    parser.add_argument('--output_uri', type = str, help= "Output URI of bucket")
    
    parser.add_argument('--experiment_id', type = str, help= "Unique id for each experiment")
    # Hyperparam
    parser.add_argument('--model_architecture', type = str, help= "Model model_architecture")
    parser.add_argument('--loss_function', type = str, help= "Loss fucntion to use")
#    parser.add_argument('--multi_gpu', type=bool, default = False, help= "Use multi gpu for training")
    parser.add_argument('--image_aug_type', type=str, default = None, help= "Type of image augmentation")
    parser.add_argument('--model_input_size', type=tuple, default = (320,320,3), help= "Model input size in (w,h,3)")
    parser.add_argument('--batch_size', type=int, default = 32, help= "Batch size")
    parser.add_argument('--epochs', type=int, default = 50, help= "Epoch")
    parser.add_argument('--learning_rate', type=float, default = 0.0001, help= "Learning rate")
    
    parser.add_argument('--lr_start', type=float, default = 1e-3, help= "lr_start")
    parser.add_argument('--lr_min', type=float, default = 1e-7, help= "lr_min")
    parser.add_argument('--lr_max', type=float, default = 1e-2, help= "lr_max")
    parser.add_argument('--warmup_epochs', type=int, default = 3, help= "warmup_epochs")
    parser.add_argument('--lr_sustain_epochs', type=int, default = 0, help= "lr_sustain_epochs")
    parser.add_argument('--lr_decay', type=float, default = 0.8, help= "lr_decay")

    parser.add_argument('--patience', type=int, default = 10, help= "Patience of early stopping")
    
    parser.add_argument('--output_model_ckpt_URI_file',type=str,
                   help = 'Path of a local txt file containing GCS path of model checkpoint for task downstreaming')
    
    args = parser.parse_args()
    
    assert args.model_architecture in ['Unet','FPN'], 'Only Unet or FPN is available for model architecture, check your input'
    assert args.loss_function in ['focal','BCE'], 'Only focal loss or BCE is available for loss function, check your input'
    assert args.image_aug_type in ['None','Geometric','Geometric + color'], 'Only None, Geometric or Geometric + color is available for image_aug_type, check your input'
    dataset = Unet_TFrecord_Dataset(train_tfrec_URI = args.train_tfrec_uri,
                                    valid_tfrec_URI = args.valid_tfrec_uri,
                                    batch_size = args.batch_size, 
                                    input_size = args.model_input_size, 
#                                    multi_gpu = args.multi_gpu,
                                    image_aug_type = args.image_aug_type)

    
    train_ds,valid_ds = dataset.prepare_dataset()
    
    LR = args.learning_rate
    EPOCHS = args.epochs
    
#     if args.multi_gpu:
#         strategy = tf.distribute.MirroredStrategy()
#         with strategy.scope():
#             model = build_unet_model()
#     else:
    model = build_unet_model(model_architecture = args.model_architecture,
                             loss_function = args.loss_function,
                             model_input_shape = (320,320,3),
                             lr = LR , 
                             weights = None)

    ######################### Callbacks##################################

    def lr_schedule(epoch):
        if epoch < args.warmup_epochs:
            lr = (args.lr_max - args.lr_start) / args.warmup_epochs * epoch + args.lr_start
        elif epoch < args.warmup_epochs + args.lr_sustain_epochs:
            lr = args.lr_max
        else:
            lr = (args.lr_max - args.lr_min) * args.lr_decay**(epoch - args.warmup_epochs - args.lr_sustain_epochs) + args.lr_min
        return lr
#     def lr_schedule(epoch, lr):
#         if epoch < 5:
#             return lr
#         else:
#             return lr * tf.math.exp(-0.2)

    tb_path = os.path.join(args.output_uri,f"tb_logs/id_{args.experiment_id}") 
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_path)

    checkpoint_dir =  '/tmp/checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True) # checkpointer does not create dir directly

    checkpoint_path = os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_loss:.2f}.h5')
    
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    patience=10, 
                                    restore_best_weights=True)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', 
                                    filepath=checkpoint_path, 
                                    save_best_only=True)
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    callbacks = [tb,
                early_stopper,
                checkpointer,
                lr_scheduler]
    ########################################################################
    
    history = model.fit(
                        train_ds, 
                        epochs=EPOCHS, 
                        steps_per_epoch=dataset.train_steps, 
                        callbacks=callbacks, 
                        validation_data = valid_ds,
                        validation_steps=dataset.valid_steps,
                        verbose=2
                        )
    
    list_of_checkpoints = glob.glob(f'{checkpoint_dir}/*.h5')
    latest_checkpoint = max(list_of_checkpoints, key=os.path.getctime)
    
    GCS_checkpoint_uri = os.path.join(args.output_uri,'checkpoint',args.experiment_id,f'{Path(latest_checkpoint).name}')
    tf.io.gfile.copy(latest_checkpoint, GCS_checkpoint_uri,overwrite=True)
    tf.io.gfile.rmtree('/tmp')
    
    Path(args.output_model_ckpt_URI_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_ckpt_URI_file).write_text(GCS_checkpoint_uri)
    
    
    metadata = {
    'outputs' : [
    # Markdown that is hardcoded inline
    {
      'storage': 'inline',
      'source': f'# Checkpoint uri\n {GCS_checkpoint_uri}',
      'type': 'markdown',
    }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    