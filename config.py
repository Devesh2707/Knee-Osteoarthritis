import albumentations
from albumentations.pytorch import ToTensorV2

class CONFIG():

    train_images = "" #path to folder where train images are stored
    test_images = "" #path to folder where test images are stored

    trainset = "dataframe/train_set_folds.csv" #csv while containing image path with label

    model_folder = "model_weights/" #path to folder where model_weights will be saved

    saliency_map = "saliency_maps/" #path where saliency will be saved

    model_name = "densenet121" #name of model used
    pretrained = True

    num_classes = 5

    virtual_batch_size = 32 #for gradient accumulation
    batch_size_train = 32
    batch_size_val = 16
    use_preprocess = True

    max_epochs = 30 
    lr = 1e-4
    weight_decay = 1e-3  #for AdamW

    nfolds = 5 #number of folds in the csv

    train_aug = albumentations.Compose([
                    albumentations.Rotate(limit = 15, p = 0.3),
                    albumentations.OneOf([
                        albumentations.Cutout(num_holes=8, max_h_size=2, max_w_size=2, fill_value=0, always_apply=False),
                        albumentations.ElasticTransform()
                    ], p = 0.5),
                    albumentations.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ), ToTensorV2(),
                ])

    val_aug = albumentations.Compose([
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                ), ToTensorV2(),
            ])

    weights_dict = {
        "resnext50_32x4d": "1-3xy7U2HwNDjUj4_3b6PqzFf9auxTM0U",
        "densenet121": "10UltQd-x3aDtoGqYNHjH3YNM35H7zRag",
        "densenet161": "1-qsVA1ZX-UdR0yPWK9V2j_Z1PP2JoYOh"
    }

