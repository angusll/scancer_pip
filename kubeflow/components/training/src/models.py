import tensorflow as tf
import segmentation_models as sm
sm.set_framework('tf.keras')

def build_unet_model(
                    model_architecture = 'Unet',
                    backbone = 'inceptionresnetv2',
                    loss_function = 'focal',
                    num_class = 1,
                    model_input_shape = (320,320,3),
                    lr = 0.001, 
                    weights = None):
    
    assert model_architecture in ['Unet','FPN'], 'Only Unet or FPN is available, check your input'
    assert loss_function in ['focal','BCE'], 'Only focal loss or BCE is available, check your input'
    
    optim = tf.keras.optimizers.Adam(lr)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    if num_class == 1:
        activation = 'sigmoid' 
        if loss_function == 'focal':
            loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 
        if loss_function == 'BCE':
            loss = tf.keras.losses.BinaryCrossentropy()
    else:
        activation = 'softmax'
        if loss_function == 'focal':
            loss = sm.losses.categorical_focal_dice_loss 
        if loss_function == 'BCE':
            loss = tf.keras.losses.BinaryCrossentropy()
        
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    #create model
    
        
    if model_architecture == 'Unet':
        model = sm.Unet(backbone, 
                        input_shape=model_input_shape,
                        weights=None,
                        encoder_weights= None,
                        classes=num_class, 
                        activation=activation)
        
    elif model_architecture == 'FPN':
        model = sm.FPN(backbone, 
                        input_shape=model_input_shape,
                        weights=None,
                        encoder_weights= None,
                        classes=num_class, 
                        activation=activation)
            
        
        
    if weights:
        model.load_weights(weights)
        
    model.compile(optim, loss, metrics)

    return model