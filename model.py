import timm
import torch.nn as nn

from config import CONFIG

cfg = CONFIG()

class CNN_Classifier(nn.Module):
    def __init__(self, model_name, pretrained = True):
        super().__init__()
        self.cnn = timm.create_model(model_name = model_name, pretrained=pretrained)
        self.classifier = nn.Sequential(nn.Dropout(0.1),
                                    nn.LeakyReLU(),
                                    nn.Linear(1000,cfg.num_classes))
        
    def forward(self, x):
        output = self.cnn(x)
        output = self.classifier(output)
        return output