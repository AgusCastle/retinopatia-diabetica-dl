import torch
import torch.nn as nn
import pdb
from models.gabor import IGConv, GaborPool
import torch.nn.functional as F


class GaborAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GaborAttentionBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gabor_filters = self.create_gabor_filters(out_channels)
        self.in_channels = in_channels
        
    def create_gabor_filters(self, num_filters):
        gabor_filters = []
        for _ in range(num_filters):
            gabor_filter = torch.randn(3, 3)  # Replace with actual Gabor filter creation
            gabor_filters.append(gabor_filter)
        return nn.ParameterList([nn.Parameter(gf) for gf in gabor_filters])
        
    def forward(self, x):
        #x = self.conv(x)
        attention_weights = torch.stack([F.conv2d(x, gf.view(1, 1, 3, 3), padding=1,) for i, gf in enumerate(self.gabor_filters)], dim=1)
        
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_weights = attention_weights.reshape(attention_weights.size(0),
                                                    attention_weights.size(1),
                                                    attention_weights.size(3),
                                                    attention_weights.size(4))
        mul = attention_weights * x
        print(mul.size())
        attended_features = torch.sum(attention_weights * x, dim=1, keepdim=True)
        print(attended_features.size())
        return mul

class AttentionLayerGabor(nn.Module):
    """
    Helper Function for real-valued gabor attention modules
    """
    def __init__(self, in_ch, no_g, gp=None):
        super().__init__()

        self.attn = nn.Sequential(
            IGConv(in_ch * 2, in_ch * no_g, kernel_size=3, padding=1, no_g=no_g),
            nn.PReLU(),
            GAMReal_Module(no_g),
            GaborPool(no_g, gp),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)
    
class GAMReal_Module(nn.Module):
    """
    Gabor attention module
    """
    def __init__(self, no_g):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.align = nn.Softmax(dim=-1)
        self.no_g = no_g

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X G X H X W)
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.size()
        C = C // self.no_g
        proj_query = x.view(m_batchsize, self.no_g, -1)
        proj_key = x.view(m_batchsize, self.no_g, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.align(energy)
        proj_value = x.view(m_batchsize, self.no_g, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C * self.no_g, height, width)

        out = self.gamma * out + x
        return out

if __name__ == '__main__':
     
     tensor = torch.randn((4, 320, 32, 32))
     #tensor = tensor.view(4, 3, -1)
     print(tensor.size())
     block = GAMReal_Module(2)
     pred = block(tensor)
     print(pred.size())

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
        self.gabor = GAMReal_Module(in_planes)
        self.cab = CAB(in_planes, n_class, k)
    
    def forward(self, x):
        x = self.gabor(x)
        x = self.gab(x)
        x = self.cab(x)
        return x


class AttnCABfc(nn.Module):
    def __init__(self, in_planes, n_class, k=5, mode='custom'):
        super(AttnCABfc, self).__init__()
        # self.reduction = nn.Sequential(nn.Conv2d(in_planes, REDUCCION , 1, padding='same'))
        self.gab_ = GAB(in_planes)
        self.gabor = GAMReal_Module(in_planes)
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
                nn.LogSoftmax(dim=1))
        else:
            self.fc_ = nn.Sequential(
                nn.Linear(in_planes, n_class),
                nn.Softmax(dim=1))

    def forward(self, x):
        x = self.gabor(x)
        x = self.gab_(x)        # torch.Size([2, 2048, 7, 7])
        x = self.cab_(x)        # torch.Size([2, 2048, 7, 7])
        x = self.avg_pool_(x)    # torch.Size([2, 2048, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([2, 2048])
        x = self.fc_(x)          # torch.Size([2, 5])

        return x


def _attn(input_feats, classes, k=5):
    module = AttnCABfc(input_feats.size(1), classes, k)
    return module
