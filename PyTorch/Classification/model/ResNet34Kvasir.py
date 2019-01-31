import torchvision.models as torchmodels
from torchvision.models.resnet import BasicBlock

class ResNet34Kvasir(torchmodels.ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3],num_classes=8)
