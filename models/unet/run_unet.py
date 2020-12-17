"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0,'/home/aditomer/baseline-wp')

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.args import Args
from common.utils import save_reconstructions
from data import transforms
from data.mri_data import SliceData
from models.unet.unet_model import UnetModel


class DataTransform:
    """
    Data Transformer for running U-Net models on a test dataset.
    """

    def __init__(self, resolution, which_challenge):
        """
        Args:
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.resolution = resolution
        self.which_challenge = which_challenge

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.Array): k-space measurements
            target (numpy.Array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (pathlib.Path): Path to the input file
            slice (int): Serial number of the slice
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Normalized zero-filled input image
                mean (float): Mean of the zero-filled image
                std (float): Standard deviation of the zero-filled image
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        
        image = transforms.complex_abs(image)
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)
        kspace = transforms.rfft2(image)
        return kspace, mean, std, fname, slice


def create_data_loaders(args):
    data = SliceData(
        root=args.data_path / f'{args.challenge}_{args.data_split}',
        transform=DataTransform(args.resolution, args.challenge),
        sample_rate=1.,
        challenge=args.challenge
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        acceleration=args.accelerations,
        center_fraction=args.center_fractions,
        res=args.resolution
    ).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet2(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames, slices) in data_loader:
            input = input.unsqueeze(1).to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def run_unet(wl=False,acc=None,cf=None,device=1):
    args = create_arg_parser(device).parse_args(sys.argv[1:])
    if acc: args.accelerations=acc
    if cf: args.center_fractions=cf
    if wl:
        args.out_dir=pathlib.Path(f'{device}rec_with')
    
    args.checkpoint=pathlib.Path(f'{acc[0]}_{cf[0]}_{wl}/best_model.pt')    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet2(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser(device):
    parser = Args()
    parser.add_argument('--data-split', choices=['val', 'test'],default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path,default=f'{device}checkpoint/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default=f'{device}rec_without', 
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')

    return parser


if __name__ == '__main__':
    run_unet()
