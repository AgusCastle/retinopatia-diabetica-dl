import torch
import torch.nn as nn
import pdb


class GAB(nn.Module):
    def __init__(self, in_planes):
        super(GAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(in_planes, in_planes, 1, padding='same'))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):      # torch.Size([2, 2048, 7, 7])
        x = self.avg_pool(inputs)   # torch.Size([2, 2048, 1, 1])
        x = self.conv2(x)           # torch.Size([2, 2048, 1, 1])
        x = self.sigmoid(x)         # torch.Size([2, 2048, 1, 1])
        C_A = x * inputs            # torch.Size([2, 2048, 7, 7])

        x = torch.mean(C_A, dim=1, keepdim=True)    # torch.Size([2, 1, 7, 7])
        x = self.sigmoid(x)         # torch.Size([2, 1, 7, 7])
        S_A = x * C_A               # torch.Size([2, 2048, 7, 7])
        return S_A


class CAB(nn.Module):
    def __init__(self, in_planes, classes, k):
        super(CAB, self).__init__()
        self.classes = classes
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, k*classes, 1, padding='same'),
                                   nn.BatchNorm2d(k*classes),
                                   nn.ReLU())
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):          # torch.Size([2, 2048, 7, 7])
        F1 = self.conv1(inputs)         # torch.Size([2, 30, 7, 7])

        F2 = F1.clone().detach().requires_grad_(True)
        GMP = self.max_pool(F2)         # torch.Size([2, 30, 1, 1])

        x = GMP.reshape([inputs.size(0), self.k, self.classes, GMP.size(
            2), GMP.size(3)])   # torch.Size([2, 6, 5, 1, 1])
        S = torch.mean(x, dim=1, keepdim=False)  # torch.Size([2, 5, 1, 1])

        x = F1.reshape([inputs.size(0), self.k, self.classes, inputs.size(
            2), inputs.size(3)])  # torch.Size([2, 6, 5, 7, 7])
        # x = F1
        x = torch.mean(x, dim=1, keepdim=False)  # torch.Size([2, 5, 7, 7])

        x = S * x                               # torch.Size([2, 5, 7, 7])
        M = torch.mean(x, dim=1, keepdim=True)  # torch.Size([2, 1, 7, 7])

        semantic = inputs * M                   # torch.Size([2, 2048, 7, 7])
        return semantic


class BlockAttencionCAB(nn.Module):
    def __init__(self, in_planes, n_class, k=5):
        super(BlockAttencionCAB, self).__init__()

        self.gab = GAB(in_planes)
        self.cab = CAB(in_planes, n_class, k)
    
    def forward(self, x):
        x = self.gab(x)
        x = self.cab(x)
        return x


class AttnCABfc(nn.Module):
    def __init__(self, in_planes, n_class, k=5, mode='custom'):
        super(AttnCABfc, self).__init__()
        # self.reduction = nn.Sequential(nn.Conv2d(in_planes, REDUCCION , 1, padding='same'))
        self.gab_ = GAB(in_planes)
        self.cab_ = CAB(in_planes, n_class, k)
        self.avg_pool_ = nn.AdaptiveAvgPool2d(1)

        if mode == 'custom':
            self.fc_ = nn.Sequential(
                nn.Linear(in_planes, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, n_class),
                nn.Softmax(dim=1))
        else:
            self.fc_ = nn.Sequential(
                nn.Linear(in_planes, n_class),
                nn.Softmax(dim=1))

    def forward(self, x):
        x = self.gab_(x)        # torch.Size([2, 2048, 7, 7])
        x = self.cab_(x)        # torch.Size([2, 2048, 7, 7])
        x = self.avg_pool_(x)    # torch.Size([2, 2048, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([2, 2048])
        x = self.fc_(x)          # torch.Size([2, 5])

        return x


def _attn(input_feats, classes, k=5):
    module = AttnCABfc(input_feats.size(1), classes, k)
    return module
