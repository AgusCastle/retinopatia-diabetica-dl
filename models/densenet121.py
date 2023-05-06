from torchvision.models import densenet121
import torch as nn

def denseNet121_agus(classes = 5):

    model = densenet121(pretrained=True, progress=True)
    n_inputs = model.classifier.in_features
    
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

    model.classifier = sequential_layers
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

denseNet121_agus()