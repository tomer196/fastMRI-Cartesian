"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import sys
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
sys.path.insert(0,'/home/aditomer/baseline-wp')

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import h5py

from common.args import Args
from common.evaluate import psnr,ssim1,nmse
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
        root=args.origin_file,
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

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def run_unet2(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    args.out_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for (input, mean, std, fnames, slices) in data_loader:
            target=[]
            for i in range(input.shape[0]):
                tmp=transforms.irfft2(input[i])
                target.append((tmp * std[i] + mean[i]).cpu().numpy())
            input = input.unsqueeze(1).to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))
                path=pathlib.Path(f'{args.out_dir}/{slices[i]}.png')
                plt.imsave(path,recons[i],cmap='gray')
                
                print(f'file: {fnames[i]}, Slice: {slices[i]}, PSNR: {psnr(recons[i].cpu().numpy(),target[i])}, SSIM: {ssim1(recons[i].cpu().numpy(),target[i])}, NMSE: {nmse(recons[i].cpu().numpy(),target[i])}')

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def save_image():
    args = create_arg_parser().parse_args(sys.argv[1:])
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet2(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = Args()
    exp_dir="/home/aditomer/baseline-wp/4_1_False/"
    parser.add_argument('--origin_file', type=pathlib.Path,default=f'/home/aditomer/Datasets/2',
                        help='Path to the U-Net model')
    parser.add_argument('--checkpoint', type=pathlib.Path,default=f'{exp_dir}best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default=f'{exp_dir}/rec', 
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')

    return parser


if __name__ == '__main__':
    save_image()
