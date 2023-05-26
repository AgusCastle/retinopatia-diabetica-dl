from torchvision.models import resnet50
from models.attentionblocks import AttnCABfc
import torch.nn as nn
import pdb


class ResNet50AB(nn.Module):
    def __init__(self, in_planes=1024, classes=5, k=5, modo='original'):
        super(ResNet50AB, self).__init__()
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])
        self.resize = nn.Conv2d(2048, in_planes, kernel_size=(
            1, 1), stride=(1, 1), bias=True)
        self.attnblocks = AttnCABfc(in_planes, classes, k, mode=modo)

    def forward(self, x):
        x = self.backbone(x)
        x = self.resize(x)
        x = self.attnblocks(x)

        return x


def resNet50Legacy(n_class):
    model = resnet50(pretrained=True, progress=True)

    n_inputs = model.fc.in_features

    sequential_layers = nn.Sequential(
        nn.Linear(n_inputs, n_class),
        nn.Softmax(dim=1)
    )

    model.fc = sequential_layers

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def resNet50Custom(n_class):
    model = resnet50(pretrained=True, progress=True)

    n_inputs = model.fc.in_features

    sequential_layers = nn.Sequential(
        nn.Linear(n_inputs, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, n_class),
        nn.LogSoftmax(dim=1)
    )

    model.fc = sequential_layers
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(count_parameters(ResNet50AB(modo='custom')))
