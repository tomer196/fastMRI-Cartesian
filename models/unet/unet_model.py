"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):
    
    def initilaize_random_mask(self,num_cols,acceleration,center_fraction):
        rng = np.random.RandomState()
        num_low_freqs = int(round(num_cols * center_fraction))
        num_high_freq = int(num_cols / acceleration) - num_low_freqs
        high_freq_mask = rng.uniform(size=(num_cols - num_low_freqs))
        sorted = np.sort(high_freq_mask)
        threshold = sorted[-num_high_freq]
        high_freq_mask[high_freq_mask >= threshold] = 1
        high_freq_mask[high_freq_mask < threshold] = 0
        pad = (num_cols - num_low_freqs + 1) // 2
        low_freq_mask = np.ones(num_low_freqs)
        mask = np.concatenate((high_freq_mask[:pad], low_freq_mask, high_freq_mask[pad:]))
        #num_high_freq = int(num_cols / acceleration)
        #high_freq_mask = rng.uniform(size=num_cols)
        #sorted = np.sort(high_freq_mask)
        #threshold = sorted[-num_high_freq]
        #high_freq_mask[high_freq_mask >= threshold] = 1
        #high_freq_mask[high_freq_mask < threshold] = 0
        #mask=high_freq_mask

        mask = mask * 0.5 + 0.5 * rng.uniform(size=num_cols)
        # sorted_parameters = np.sort(mask)
        # threshold = sorted_parameters[320 - int(320 / acceleration)]
        # bimask = mask >= threshold
        # plt.imshow(np.ones((320,320))*bimask)
        # plt.show()
        # print(bimask)
        return torch.tensor(mask,dtype=torch.float)


    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,acceleration,center_fraction,res):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.acceleration=acceleration[0]
        self.center_fraction=center_fraction[0]
        self.velocity = 0.
        self.momentum = 0.9
        self.use_random=False

        mask= self.initilaize_random_mask(res, self.acceleration,self.center_fraction)
        self.mask = torch.nn.Parameter(mask,requires_grad=False)
        mask_shape = [1,1,res,1,1]
        Bimask_1=torch.tensor(mask)
        self.Bimask = torch.nn.Parameter(torch.reshape(Bimask_1, mask_shape),requires_grad=True)

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        self.make_mask(input.shape)
        input=input*self.Bimask
        input=self.transform(input)

        #plt.imshow(input.detach().cpu().numpy()[0,0,:,:])
        #plt.show()

        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

    def apply_binary_grad(self,lr):
        self.velocity=self.momentum*self.velocity+(1-self.momentum)*self.Bimask.grad[0,0,:,0,0]        
        self.mask-=lr*self.velocity
        #with torch.no_grad():
        #  self.mask.add_(torch.randn(self.mask.size()).cuda() * 0.00001)
             
        #mean = self.mask.mean()
        #std = self.mask.std()
           
        #self.mask.data=(self.mask - mean) / std*0.25 #normalize to mean=0, std=0.25
        
        self.mask.clamp(-1,1)
        return

    def make_mask(self, shape):
        res=shape[2]
        if self.use_random:
            with torch.no_grad():
                self.mask.data = self.initilaize_random_mask(res, self.acceleration,self.center_fraction).to('cuda')
               
        sorted_parameters, _ = torch.sort(self.mask)
        threshold = sorted_parameters[res - int(res / self.acceleration)]
        with torch.no_grad():
            self.Bimask[0,0,:,0,0]=self.mask >= threshold
        return

    def transform(self, input):
        input = self.ifft(input)
        input = self.complex_abs(input)
        input = self.normalize_instance(input, eps=1e-11)
        input = input.clamp(-6, 6)
        return input

    def ifft(self, data):
        # assert data.size(-1) == 2
        # data = self.ifftshift(data, dim=(-3, -2))
        data = torch.ifft(data, 2, normalized=True)
        data = self.fftshift(data, dim=(-3, -2))
        return data

    def complex_abs(self, data):
        assert data.size(-1) == 2
        return (data ** 2).sum(dim=-1).sqrt()

    def complex_center_crop(self, data, shape):
        assert 0 < shape[0] <= data.shape[-3]
        assert 0 < shape[1] <= data.shape[-2]
        w_from = (data.shape[-3] - shape[0]) // 2
        h_from = (data.shape[-2] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to, :]

    def normalize_instance(self, data, eps=0.):
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + eps)

    def roll(self,x, shift, dim):
        if isinstance(shift, (tuple, list)):
            assert len(shift) == len(dim)
            for s, d in zip(shift, dim):
                x = self.roll(x, s, d)
            return x
        shift = shift % x.size(dim)
        if shift == 0:
            return x
        left = x.narrow(dim, 0, x.size(dim) - shift)
        right = x.narrow(dim, x.size(dim) - shift, shift)
        return torch.cat((right, left), dim=dim)

    def fftshift(self,x, dim=None):
        if dim is None:
            dim = tuple(range(x.dim()))
            shift = [dim // 2 for dim in x.shape]
        elif isinstance(dim, int):
            shift = x.shape[dim] // 2
        else:
            shift = [x.shape[i] // 2 for i in dim]
        return self.roll(x, shift, dim)

    def ifftshift(self,x, dim=None):
        if dim is None:
            dim = tuple(range(x.dim()))
            shift = [(dim + 1) // 2 for dim in x.shape]
        elif isinstance(dim, int):
            shift = (x.shape[dim] + 1) // 2
        else:
            shift = [(x.shape[i] + 1) // 2 for i in dim]
        return self.roll(x, shift, dim)
