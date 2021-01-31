import pretrainedmodels
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Resnet
class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torchvision.models.resnet34(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.out = nn.Linear(in_features, 5)

    def forward(self, image, targets=None):

        batch_size, C, H, W = image.shape
        x = self.base_model.conv1(image)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = self.out(x)
        return x


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
