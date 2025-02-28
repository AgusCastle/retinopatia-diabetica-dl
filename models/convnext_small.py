from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import ConvNormActivation
from torchvision._internally_replaced_utils import load_state_dict_from_url

from models.attentionblocks import BlockAttencionCAB, AttnCABfc

class ConvNeXtSmall(nn.Module):
    def __init__(self, classes=5, attn = [True, True, True, True], gabor = 0, no_g = 1) -> None:
        super().__init__()
        self.layer_scale = 1e-6
        self.n_layers = [3, 3, 27, 3]
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        prob_sto = 0.011428571428571429
        count_blocks = 0
        
        layers = []
        features = []
        features.append(ConvNormActivation(3, 96, kernel_size=4, 
                                                    stride=4, 
                                                    padding=0,
                                                    norm_layer=norm_layer,
                                                    activation_layer=None,
                                                    bias=True))

        # Bloque 1 [3, 96]

        for i in range(self.n_layers[0]):
            layers.append((CNBlock(96, self.layer_scale, prob_sto * count_blocks)))
            count_blocks += 1

        if attn[0]:
            layers.append(BlockAttencionCAB(in_planes=96, n_class= 5, gabor=gabor, no_g=no_g))

        features.append(nn.Sequential(*layers))

        # DownSampling 96 -> 192
        features.append(nn.Sequential(
                            norm_layer(96),
                            nn.Conv2d(96, 192, kernel_size=2, stride=2),
                        ))

        # Bloque [3, 192]

        layers = []
        for i in range(self.n_layers[1]):
            layers.append((CNBlock(192, self.layer_scale, prob_sto * count_blocks)))
            count_blocks += 1
        
        if attn[1]:
            layers.append(BlockAttencionCAB(in_planes=192, n_class= 5, gabor=gabor, no_g=no_g))

        features.append(nn.Sequential(*layers))

        # DownSampling 192 -> 384

        features.append(nn.Sequential( 
                                      norm_layer(192),
                                      nn.Conv2d(192, 384, kernel_size=2, stride=2)))

        # Bloque [27, 384]

        layers = []
        for i in range(self.n_layers[2]):
            layers.append((CNBlock(384, self.layer_scale, prob_sto * count_blocks)))
            count_blocks += 1
        
        if attn[2]:
            layers.append(BlockAttencionCAB(in_planes=384, n_class= 5, gabor=gabor, no_g=no_g))

        features.append(nn.Sequential(*layers))

        # DownSampling 384 -> 768

        features.append(nn.Sequential(
                                      norm_layer(384),
                                      nn.Conv2d(384, 768, kernel_size=2, stride=2)))
        
        # Bloque [3, 768]

        layers = []
        for i in range(self.n_layers[3]):
            if i == 1:
                layers.append((CNBlock(768, self.layer_scale, 0.3885714285714286)))
            else:
                layers.append((CNBlock(768, self.layer_scale, prob_sto * count_blocks)))
            count_blocks += 1

        features.append(nn.Sequential(*layers))

        self.features = nn.Sequential(*features)

        if attn[3]:
            self.attb = AttnCABfc(in_planes=768, n_class=classes, k=5, mode='custom', gabor=gabor, no_g=no_g)
        else:
            self.attb  = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(768, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, classes),
                nn.LogSoftmax(dim=1))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.attb(x)
        return x

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)

class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _convnext_small(classes = 1000, pretrained = True, b_attn = [False, False, False, False], gabor = 0, no_g = 1):

    model = ConvNeXtSmall(classes, attn = b_attn, gabor=gabor, no_g=no_g)

    if pretrained:
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/convnext_small-0c510722.pth", progress=True)
            model.load_state_dict(state_dict, strict = False)
    return model

def convnext_small(classes, pretrained = True, b_attn = [0, 0, 0, 0], gabor = 0, no_g = 1):

    b_attn = [bool(x) for x in b_attn]

    model = _convnext_small(classes=classes, pretrained=pretrained, b_attn=b_attn, gabor=gabor, no_g=no_g)
    
    return model

if __name__ == '__main__':
    print(count_parameters(convnext_small(5)))