import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from trainer import CNNModule
from config import CONFIG

from utils import download_weights, scaling, preprocess_image, preprocess_salient

cfg = CONFIG()

def superimpose(saliency, img):
    img_array = np.asarray(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = np.uint8(scaling(img_array, 255, 0))
    s = saliency[0]
    s = np.uint8(scaling(s, 255, 0))
    superimposed = cv2.addWeighted(s, 0.8, img_array, 0.2, 0)
    return superimposed


def create_saliency_map(model_name, img_tensor):
    model = CNNModule(model_name, pretrained = False)
    weights = torch.load(f'{cfg.model_folder}{model_name}.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(weights['state_dict'])
    model = model.net

    model.eval()

    img_tensor.requires_grad_()

    scores = model(img_tensor)

    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]
    score_max.backward()

    saliency, _ = torch.max(img_tensor.grad.data.abs(),dim=1)

    del model

    return saliency, torch.sigmoid(scores)

def make_saliency_map(img, model_name):
    img = Image.fromarray(preprocess_image(img))
    saliency, scores = create_saliency_map(model_name, preprocess_salient(img))
    superimposed_image = superimpose(saliency, img)
    plt.imshow(superimposed_image, cmap = 'hot')
    plt.axis('off')
    plt.savefig(f'{cfg.saliency_map}map_hm.jpg', bbox_inches='tight')
    plt.imshow(superimposed_image)
    plt.axis('off')
    plt.savefig(f'{cfg.saliency_map}map.jpg', bbox_inches='tight')
    return scores
    