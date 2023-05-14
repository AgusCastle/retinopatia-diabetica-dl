from torchvision.models import resnext50_32x4d
import torch.nn as nn

def resNeXt50_agus(classes = 5):
    
    model = resnext50_32x4d(pretrained=True, progress=True)
    
    n_inputs = model.fc.in_features

    sequential_layers = nn.Sequential(
        nn.Linear(n_inputs, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, classes),
        nn.Softmax(dim=1)
    )

    model.fc = sequential_layers
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
