import pretrainedmodels
import torch.nn as nn

# Resnet
def get_model(pretrained):
    if pretrained:
        model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
    else:
        model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=512, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024, eps=1e-5, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=5),
        nn.Sigmoid(),
    )
    return model


# Efficientnet
from efficientnet_pytorch import EfficientNet


class LeafModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained("efficientnet-b3")
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1536, num_classes)

    def forward(self, image):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))
        return outputs
