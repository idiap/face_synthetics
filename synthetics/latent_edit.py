#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# Adapted from:
# https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_synthetic_dataset/
# This code was entirely written by a human

import os

import click
import pickle as pkl
from dataclasses import dataclass

import torch

from . import utils
from .utils import Sample

# ---

class LatentEdit():
    EDIT_ITEMS = ['smile', 'scream', 'disgust', 'squint', 'surprise']

    @dataclass
    class CovariantScaling:
        pose: float = 1.0
        illumination : float = 1.0
        expression : float = 1.0

    # ---

    def __init__(
            self,
            network_type : str,
            device = torch.device('cuda'),
            dtype = torch.float32) -> None:
        assert network_type in utils.network_types()
        self.network_type = network_type
        self.device = device
        self.dtype = dtype
        self.covariates_analysis = None

    # ---

    def load_covariates_analysis(
            self,
            path : str = None):
        if path is None:
            path = os.path.join(utils.config['latent_directions'], self.network_type + '.pkl')
        try:
            with open(path, 'rb') as f:
                self.covariates_analysis = pkl.load(f)
        except:
            raise RuntimeError(f'Cannot load latent direction pickle file: {path}')
        try:
            self.pose_normal = torch.tensor(
                self.covariates_analysis['pose']['normal'],
                device=self.device,
                dtype=self.dtype)
            self.pose_neg_mean = torch.tensor(
                self.covariates_analysis['pose']['neg_stats']['mean'],
                device=self.device,
                dtype=self.dtype)
            self.pose_pos_mean = torch.tensor(
                self.covariates_analysis['pose']['pos_stats']['mean'],
                device=self.device,
                dtype=self.dtype)
        except:
            click.echo('WARNING: Cannot load pose analysis...')
            self.pose_normal = None
            self.pose_neg_mean = None
            self.pose_pos_mean = None
        try:
            self.illumination_normal = torch.tensor(
                self.covariates_analysis['illumination']['normal'],
                device=self.device,
                dtype=self.dtype)
            self.illumination_neg_mean = torch.tensor(
                self.covariates_analysis['illumination']['neg_stats']['mean'],
                device=self.device,
                dtype=self.dtype)
            self.illumination_pos_mean = torch.tensor(
                self.covariates_analysis['illumination']['pos_stats']['mean'],
                device=self.device,
                dtype=self.dtype)
        except:
            click.echo('WARNING: Cannot load illumination analysis...')
            self.illumination_normal = None
            self.illumination_neg_mean = None
            self.illumination_pos_mean = None

        self.expression_items = []
        self.expression_normal = {}
        self.expression_neg_mean = {}
        self.expression_pos_mean = {}
        try:
            for _, expr in self.covariates_analysis['expression'].keys():
                self.expression_items.append(expr)
                self.expression_normal[expr] = torch.tensor(
                    self.covariates_analysis['expression'][('neutral', expr)]['normal'],
                    device=self.device,
                    dtype=self.dtype)
                self.expression_neg_mean[expr] = torch.tensor(
                    self.covariates_analysis['expression'][('neutral', expr)]['neg_stats']['mean'],
                    device=self.device,
                    dtype=self.dtype)
                self.expression_pos_mean[expr] = torch.tensor(
                    self.covariates_analysis['expression'][('neutral', expr)]['pos_stats']['mean'],
                    device=self.device,
                    dtype=self.dtype)
        except:
            click.echo('WARNING: Cannot load expression analysis...')
            self.expression_items = None
            self.expression_normal = None
            self.expression_neg_mean = None
            self.expression_pos_mean = None

    # ---

    def neutralisation(
            self,
            w: torch.Tensor):
        assert self.covariates_analysis is not None
        # Pose neutralisation : project on binary decision boundary
        if self.pose_normal is not None:
            dot = torch.einsum('bl, il -> b', w, self.pose_normal)
            w -= torch.einsum('b, il -> bl', dot, self.pose_normal)
        # Illumination neutralisation : project on binary decision boundary
        if self.illumination_normal is not None:
            dot = torch.einsum('bl, il -> b', w, self.illumination_normal)
            w -= torch.einsum('b, il -> bl', dot, self.illumination_normal)
        # Expression neutralisation
        if self.expression_normal is not None:
            ### Assumption : default sampled latents are always either 'neutral' or 'smile'
            # 1. Cancel component
            dot = torch.einsum('bl, il -> b', w, self.expression_normal['smile'])
            w -= torch.einsum('b, il -> bl', dot, self.expression_normal['smile'])
            # 2. Move towards neutral using the mean distance computed on train set
            w += self.expression_neg_mean['smile'] * \
                self.expression_normal['smile'].repeat(w.shape[0], 1)
        return w

    # ---

    def binary_augmentation(
            self,
            w : torch.Tensor,
            neg_mean : torch.Tensor,
            pos_mean : torch.Tensor,
            normal : torch.Tensor,
            num_latents : int,
            scaling : float):
        assert w.shape[0] == 1, 'batch mode not supported'
        extremum = max(-neg_mean, pos_mean)
        scales = scaling * torch.linspace(-extremum, extremum, num_latents, device=self.device)
        scaled_normals = torch.einsum('s, il -> sl', scales, normal)
        w_augmentations = w.repeat(num_latents, 1) + scaled_normals
        return w_augmentations, scales

    # ---

    def pose_augmentation(
            self,
            w : torch.Tensor,
            num_latents : int,
            scaling : float = 3/4):
        assert self.covariates_analysis is not None
        if self.pose_normal is not None:
            return self.binary_augmentation(
                w,
                self.pose_neg_mean,
                self.pose_pos_mean,
                self.pose_normal,
                num_latents,
                scaling)
        else:
            return w, None

    # ---

    def illumination_augmentation(
            self,
            w : torch.Tensor,
            num_latents : int,
            scaling : float = 5/4):
        assert self.covariates_analysis is not None
        if self.illumination_normal is not None:
            return self.binary_augmentation(
                w,
                self.illumination_neg_mean,
                self.illumination_pos_mean,
                self.illumination_normal,
                num_latents,
                scaling)
        else:
            return w, None

    # ---

    def expression_augmentation(
            self,
            w : torch.Tensor,
            scaling : float = 3/4):
        assert self.covariates_analysis is not None
        assert w.shape[0] == 1, 'batch mode not supported'
        assert self.expression_normal is not None
        new_latents = []
        for direction in self.expression_items:
            # 1. Cancel neutral component
            dot = torch.einsum('bl, il -> b', w, self.expression_normal[direction])
            w_augmented = w - torch.einsum('b, il -> bl', dot, self.expression_normal[direction])
            # 2. Move towards expression using the mean distance
            #    computed on train set to ensure realistic outcome
            w_augmented += scaling * self.expression_pos_mean[direction] * self.expression_normal[direction]
            new_latents.append(w_augmented)
        new_latents = torch.cat(new_latents)
        return new_latents, self.expression_items

    # ---

    def augmentation(
            self,
            w : torch.Tensor,
            covariates_scaling : CovariantScaling = CovariantScaling(),
            append_original : bool = True):
        w_pose, pose_labels = self.pose_augmentation(
            w,
            num_latents=6,
            scaling=covariates_scaling.pose)
        pose_labels = ['pose_{}'.format(item) for item in range(len(pose_labels))]
        w_illumination, illumination_labels = self.illumination_augmentation(
            w,
            num_latents=6,
            scaling=covariates_scaling.illumination)
        illumination_labels = ['illumination_{}'.format(item) for item in range(len(illumination_labels))]
        w_expression, expression_labels = self.expression_augmentation(
            w,
            scaling=covariates_scaling.expression)
        if append_original:
            latents = torch.cat([w, w_pose, w_illumination, w_expression])
            labels = ['original'] + pose_labels + illumination_labels + expression_labels
        else:
            latents = torch.cat([w_pose, w_illumination, w_expression])
            labels = pose_labels + illumination_labels + expression_labels
        return latents, labels

# ---

@click.command(
    help='Edit an existing w-latent vector')
@click.pass_context
@click.option(
    '--network-type',
    '-nt', help='Network',
    type=click.Choice(utils.network_types()),
    default='stylegan2')
@click.option(
    '--seed',
    '-s',
    type=int,
    help='Torch seed',
    default=123,
    show_default=True)
@click.option(
    '--input',
    '-i',
    type=str,
    help='Input latent HDF5 file',
    required=True)
@click.option(
    '--neutralize',
    '-n',
    is_flag=True,
    help='Neutralize facial expression before editing')
@click.option(
    '--edit-type',
    '-e',
    help='Editing type',
    type=click.Choice(LatentEdit.EDIT_ITEMS),
    default=None)
@click.option(
    '--scaling',
    '-s',
    help='Editing scaling',
    type=float,
    default=1.0)
@click.option(
    '--analysis-path',
    '-a',
    type=str,
    help='Latent analysis pickle file, network default is not specified',
    default=None)
@click.option(
    '--out',
    '-o', type=click.Path(file_okay=True, dir_okay=False),
    help='File path to save the results',
    default='out.h5',
    show_default=True)
def  latent_edit(
        ctx: click.Context,
        network_type : str,
        seed : int,
        input : str,
        neutralize : bool,
        edit_type : str,
        scaling : float,
        analysis_path : str,
        out : str ):
    torch.manual_seed(seed)
    device = torch.device('cpu')
    editor = LatentEdit(network_type=network_type)
    editor.load_covariates_analysis(path=analysis_path)
    sample = Sample()
    sample.load(file_path=input, device=device)
    assert sample.network_type == network_type, 'incorrect network type'
    assert sample.network_type != 'eg3d', 'EG3D not (yet) supported'
    assert sample.w_latent is not None, 'w latent missing in file, w-plus not (yet) supported'
    w = sample.w_latent

    if neutralize:
        w = editor.neutralisation(w)

    if edit_type is not None:
        w, _ = editor.expression_augmentation(w, scaling=scaling)
        # TODO better
        i = LatentEdit.EDIT_ITEMS.index(edit_type)
        w = w[i].unsqueeze(0)

    sample = Sample(w_latent=w, network_type=network_type)
    sample.save(file_path=out)
