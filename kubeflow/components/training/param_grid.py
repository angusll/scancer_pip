param_grid = {0: {'image_aug_type': 'None', 'loss_function': 'focal', 'model_architecture': 'Unet'},
                1: {'image_aug_type': 'None', 'loss_function': 'focal', 'model_architecture': 'FCN'},
                2: {'image_aug_type': 'None', 'loss_function': 'BCE', 'model_architecture': 'Unet'},
                3: {'image_aug_type': 'None', 'loss_function': 'BCE', 'model_architecture': 'FCN'},
                4: {'image_aug_type': 'Geometric', 'loss_function': 'focal', 'model_architecture': 'Unet'},
                5: {'image_aug_type': 'Geometric', 'loss_function': 'focal', 'model_architecture': 'FCN'},
                6: {'image_aug_type': 'Geometric', 'loss_function': 'BCE', 'model_architecture': 'Unet'},
                7: {'image_aug_type': 'Geometric', 'loss_function': 'BCE', 'model_architecture': 'FCN'},
                8: {'image_aug_type': 'Geometric + color','loss_function': 'focal','model_architecture': 'Unet'},
                9: {'image_aug_type': 'Geometric + color','loss_function': 'focal','model_architecture': 'FCN'},
                10: {'image_aug_type': 'Geometric + color', 'loss_function': 'BCE', 'model_architecture': 'Unet'},
                11: {'image_aug_type': 'Geometric + color', 'loss_function': 'BCE', 'model_architecture': 'FCN'}}