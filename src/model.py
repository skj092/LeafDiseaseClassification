# !pip install pretrainedmodels -q 
import pretrainedmodels
import torch.nn as nn

def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__['resnet18'](
            pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__['resnet18'](
            pretrained=None)
    model.last_linear = nn.Sequential(
        nn.Linear(in_features=512, out_features=5),
        nn.Sigmoid())
    return model