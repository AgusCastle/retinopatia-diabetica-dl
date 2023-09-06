import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d


def match_shape(x, y, compress=True):
    """Reshapes a tensor to be broadcastable with another

    The input tensor, x, by default will be reshaped so that all but the first
    dimensions match all but the first dimensions of y.
    Args:
        compress (boolean): If false x will be reshaped so that it's first
            dimension is split into two, with the first matching that of y.

    Returns:
        A reshaped tensor
    """
    if compress:
        x = x.view(-1, *y.size()[1:])
    else:
        x = x.view(y.size(1), -1, *x.size()[1:])
    return x


def cartesian_coords(weight):
    """Generates cartesian coordinates for analytical filter.

    Args:
        weight: The weight to be passed through the filter. This is used to
            set compatible tensor properties, e.g. height, width, device.

    Returns:
        torch.tensor: x coordinates
        torch.tensor: y coordinates
    """
    h = weight.size(-2)
    w = weight.size(-1)
    y, x = torch.meshgrid([
        torch.arange(-h / 2, h / 2).to(weight.device),
        torch.arange(-w / 2, w / 2).to(weight.device)
    ])
    x = x + .5
    y = y + .5
    return x, y


def norm(t, eps=1e-12):
    """Normalises tensor between 0 and 1
    """
    return (t - t.min()) / (t.max() - t.min() + eps)


def f_h(x, y, sigma=math.pi, eps=1e-5):
    """First half of filter
    """
    return torch.exp(-(x ** 2 + y ** 2) / (2 * (sigma + eps) ** 2))


def s_h(x_p, lambd):
    """Second half of filter
    """
    return torch.cos(2 * math.pi / lambd * x_p)


def x_prime(x, y, theta):
    """Computes x*cos(theta) + y*sin(theta)
    """
    return (
        torch.cos(theta).unsqueeze(1).unsqueeze(1) * x
        + torch.sin(theta).unsqueeze(1).unsqueeze(1) * y
    )


def y_prime(x, y, theta):
    """Computes y*cos(theta) - x*sin(theta)
    """
    return (
        torch.cos(theta).unsqueeze(1).unsqueeze(1) * y
        - torch.sin(theta).unsqueeze(1).unsqueeze(1) * x
    )


def gabor(weight, params):
    """Computes a gabor filter.

    Args:
        weight: The weight to be passed through the filter. This is used to
            set compatible tensor properties, e.g. height, width, device.
        params: theta and sigma parameters.
            Must have params.size() = [G, 2].
            Here G = no of gabor filters.
            params[:, 0] = theta parameters.
            params[:, 1] = sigma parameters.

    Returns:
        torch.tensor: gabor filter with (F_out*G, F_in, H, W) dimensions
    """
    x, y = cartesian_coords(weight)
    theta = params[0]
    lambd = params[1].unsqueeze(1).unsqueeze(1)
    x_p = x_prime(x, y, theta)
    return norm(f_h(x, y) * s_h(x_p, lambd))


class IGabor(nn.Module):
    """Wraps the Gabor implementation into a NN layer w/o convolution.

    Args:
        no_g (int, optional): The number of desired Gabor filters.
        layer (boolean, optional): Whether this is used as a layer or a
            modulation function.
    """
    def __init__(self, no_g=4, layer=False, kernel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.gabor_params = nn.Parameter(data=torch.Tensor(2, no_g))
        self.gabor_params.data[0] = torch.arange(no_g) / (no_g) * math.pi
        self.gabor_params.data[1].uniform_(-1 / math.sqrt(no_g),
                                           1 / math.sqrt(no_g))
        self.register_parameter(name="gabor", param=self.gabor_params)
        self.register_buffer("gabor_filters", torch.Tensor(no_g, 1,
                                                           *kernel_size))

        self.no_g = no_g
        self.layer = layer

    def forward(self, x):
        # print(f'x.size()={x.unsqueeze(1).size()}, gabor={gabor(x, self.gabor_params).unsqueeze(1).size()}')
        if self.training:
            self.generate_gabor_filters(x)

        # print(f'self.gabor_filters.size()={self.gabor_filters.size()}')

        out = self.gabor_filters * x.unsqueeze(1)

        # print(f'out.size()={out.size()}')

        out = out.view(-1, *out.size()[2:])

        # print(f'out.size()={out.size()}')

        if self.layer:
            out = out.view(x.size(0), x.size(1) * self.no_g, *x.size()[2:])
        #print('IGABOR {}'.format(out.size()))
        return out

    def generate_gabor_filters(self, x):
        """Generates the gabor filter bank
        """
        self.gabor_filters = gabor(x, self.gabor_params).unsqueeze(1)


class IGConv(Conv2d):
    """Implements a convolutional layer where weights are first Gabor modulated.

    In addition, rotated pooling, gabor pooling and batch norm are implemented
    below.
    Args:
        input_features (torch.Tensor): Feature channels in.
        output_features (torch.Tensor): Feature channels out.
        kernel_size (int, tuple): Size of kernel.
        rot_pool (bool, optional):
        no_g (int, optional): The number of desired Gabor filters.
        pool_stride (int, optional):
        plot (bool, optional): Plots feature maps/weights
        max_gabor (bool, optional):
        conv_kwargs (dict, optional): Contains keyword arguments to be passed
            to convolution operator. E.g. stride, dilation, padding.
    """
    def __init__(self, input_features, output_features, kernel_size,
                 pooling=None, no_g=2, pool_stride=1, pool_kernel=3, plot=False,
                 max_gabor=False, include_gparams=False, **conv_kwargs):
        if not max_gabor:
            if output_features % no_g:
                raise ValueError("Number of filters ({}) does not divide output features ({})"
                                 .format(str(no_g), str(output_features)))
            output_features //= no_g
        kernel_size = _pair(kernel_size)
        super().__init__(input_features, output_features, kernel_size, **conv_kwargs)
        self.conv = F.conv2d

        self.gabor = IGabor(no_g, kernel_size=kernel_size)
        self.no_g = no_g
        self.pooling = None
        if pooling is not None:
            self.pooling = pooling(kernel_size=pool_kernel, stride=pool_stride)
        self.max_gabor = max_gabor
        self.conv_kwargs = conv_kwargs
        self.bn = nn.BatchNorm2d(output_features * no_g)
        self.include_gparams = include_gparams

        self.plot = None

    def forward(self, x):
        enhanced_weight = self.gabor(self.weight)
        out = self.conv(x, enhanced_weight, **self.conv_kwargs)
        out = self.bn(out)

        if self.plot is not None:
            self.plot.update(self.weight[:, 0].clone().detach().cpu().numpy(),
                             enhanced_weight[:, 0].clone().detach().cpu().numpy(),
                             self.gabor_params.clone().detach().cpu().numpy())

        if self.pooling is not None:
            out = self.pooling(out)

        if self.max_gabor or self.include_gparams:
            max_out = out.view(out.size(0),
                               enhanced_weight.size(0) // self.no_g,
                               self.no_g,
                               out.size(2),
                               out.size(3))
            max_out, max_idxs = torch.max(max_out, dim=2)

        if self.max_gabor:
            out = max_out

        if self.include_gparams:
            max_gparams = self.gabor.gabor_params[0, max_idxs]
            out = torch.stack(out, max_gparams, dim=3)

        return out


class GaborPool(nn.Module):
    """
    """
    def __init__(self, no_g, pool_type='max', **kwargs):
        super().__init__(**kwargs)
        self.no_g = no_g
        self.pool_type = pool_type
        if pool_type == 'max':
            self.pooling = torch.max
        elif pool_type == 'avg':
            self.pooling = torch.mean
        else:
            self.pooling = torch.max

    def forward(self, x):
        b, c, w, h = x.size()
        reshaped = x.view(b, c // self.no_g, self.no_g, w, h)
        out, _ = self.pooling(reshaped, dim=2)
        return out


def _pair(x):
    if np.issubdtype(type(x), np.integer) and np.isscalar(x):
        return (x, x)
    return x