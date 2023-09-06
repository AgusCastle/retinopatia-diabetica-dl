import torch
import torch.nn as nn
import pdb
from models.gabor import IGConv, GaborPool
import torch.nn.functional as F

class AttentionLayerGabor(nn.Module):
    """
    Helper Function for real-valued gabor attention modules
    """
    def __init__(self, in_ch, no_g, gp=None):
        super().__init__()

        self.attn = nn.Sequential(
            IGConv(in_ch, in_ch, kernel_size=3, padding=1, no_g=no_g),
            nn.PReLU(),
            GAMReal_Module(no_g),
            GaborPool(no_g, gp),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.attn(x)
        return x
    
    
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
    def __init__(self, in_planes, n_class, k=5, gabor = 0 ,no_g = 1):
        super(BlockAttencionCAB, self).__init__()

        self.gabor_op = gabor
        if gabor != 0:
            self.gabor = AttentionLayerGabor(in_planes, no_g)

        self.gab = GAB(in_planes)
        self.cab = CAB(in_planes, n_class, k)

    def forward_pre_gab(self, x):
        
        x = self.gabor(x)
        x = self.gab(x)
        x = self.cab(x)
        
        return x
    
    def forward_post_gab(self, x):

        x = self.gab(x)
        x = self.gabor(x)
        x = self.cab(x)
        
        return x

    def forward(self, x):

        if self.gabor_op == 2:
            x = self.forward_post_gab(x)
        elif self.gabor_op == 1:
            x = self.forward_pre_gab(x)
        else:
            x = self.gab(x)
            x = self.cab(x)
        return x


class AttnCABfc(nn.Module):
    """
    Gabor   : 0 sin atencion gabor
            : 1 antes de GAB
            : 2 despues de GAB
    """
    def __init__(self, in_planes, n_class, k=5, mode='custom', gabor = 0, no_g = 1):
        super(AttnCABfc, self).__init__()       
        self.gabor_op = gabor
        if gabor != 0:
            
            self.gabor = AttentionLayerGabor(in_planes , no_g)
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
                nn.LogSoftmax(dim=1))
        else:
            self.fc_ = nn.Sequential(
                nn.Linear(in_planes, n_class),
                nn.Softmax(dim=1))

    def forward_pre_gab(self, x):
        
        x = self.gabor(x)
        x = self.gab_(x)
        x = self.cab_(x)        
        x = self.avg_pool_(x)    
        x = torch.flatten(x, 1)  
        x = self.fc_(x)          

        return x

    def forward_post_gab(self, x):

        x = self.gab_(x)
        x = self.gabor(x)
        x = self.cab_(x)        
        x = self.avg_pool_(x)    
        x = torch.flatten(x, 1)  
        x = self.fc_(x)

        return x

    def forward(self, x):

        if self.gabor_op == 2:
            x = self.forward_post_gab(x)
        elif self.gabor_op == 1:
            x = self.forward_pre_gab(x)
        else:
            x = self.gab_(x)        # torch.Size([2, 2048, 7, 7])
            x = self.cab_(x)        # torch.Size([2, 2048, 7, 7])
            x = self.avg_pool_(x)    # torch.Size([2, 2048, 1, 1])
            x = torch.flatten(x, 1)  # torch.Size([2, 2048])
            x = self.fc_(x)          # torch.Size([2, 5])
        return x