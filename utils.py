from torchvision import transforms as T
import cv2
from google_drive_downloader import GoogleDriveDownloader as gdd
from config import CONFIG

cfg = CONFIG()

def preprocess_salient(image, size=224):
    transform = T.Compose([
        T.Resize((size,size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)

def deprocess_salient(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)

def preprocess_image(img_array):
  gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
  smooth = cv2.GaussianBlur(gray,(3,3),0)
  eql = cv2.equalizeHist(smooth)
  return cv2.cvtColor(eql, cv2.COLOR_GRAY2RGB)

  
def scaling(X, high, low):
  X_std = (X - X.min()) / (X.max() - X.min())
  X_scaled = X_std * (high - low) + low
  return X_scaled

def download_weights(model_name):
    gdd.download_file_from_google_drive(file_id=cfg.weights_dict[model_name],
                                        dest_path = f'{cfg.model_folder}{model_name}.ckpt')