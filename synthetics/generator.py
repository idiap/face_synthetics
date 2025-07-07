#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import click
from math import pi

import torch
import torchvision
import einops

from . import utils
from .utils import Sample
from .eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics

# ---

class Generator():

    def __init__(
            self,
            network_type : str = 'stylegan2',
            network_path: str | None = None,
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32,
            require_grad : bool = False):
        self.network_type = network_type
        self.device = device
        self.dtype = dtype
        self.require_grad = require_grad
        self.network = utils.get_network(
            network_type,
            network_path,
            require_grad,
            self.device)
        self.w_dim = self.network.w_dim
        self.z_dim = self.network.z_dim
        if self.network_type == 'eg3d':
            self.num_ws = self.network.backbone.num_ws
            self.w_avg = self.network.backbone.mapping.w_avg
            self.__depth_map = None
            self.__raw_image = None
        else:
            self.num_ws = self.network.num_ws
            self.w_avg = self.network.mapping.w_avg
        self.img_resolution = self.network.img_resolution

    def mapping(
            self,
            z : torch.Tensor,
            c : torch.Tensor = None,
            truncation_psi : float = 0.5,
            w_plus : bool = False,
            old_mode = False) -> torch.Tensor:
        """
            Map a sample in the \mathcal{Z} latent space (B, 512) to
            the \mathcal{W} latent space (B, 512). If the ``w_plus``
            flag is set to ``True`` the resulting vector has shape
            (B, num_ws, 512).
        """
        assert type(z) == torch.Tensor
        assert z.ndim == 2 and z.shape[1] == self.z_dim
        if c is not None:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 2 and c.shape[0] == z.shape[0] and c.shape[1] == 25
        if self.network_type == 'eg3d':
            assert c is not None
        if old_mode:
            # TODO check difference
            w = self.network.mapping(z, c)
            w = self.w_avg + (self.w_avg - w) * truncation_psi
        else:
            w = self.network.mapping(z, c, truncation_psi=truncation_psi)
        if not w_plus:
            w = w[:,0,:]
        return w

    def synthesis(
            self,
            w : torch.Tensor,
            c : torch.Tensor = None,
            w_plus : bool = False,
            noise_mode : str = 'random') -> torch.Tensor:
        """
            Synthesize an image (B, 3, H, W) from a latent vector (B, 512).
            If the w_plus flag is set to ``True`` the latent must be
            of shape (B, num_ws, 512). For the EG3D network, if the
            ``return_raw_depth`` parameter is set to ``True`` the function
            return a tuple of images (image, image_raw, depth).
        """
        assert type(w) == torch.Tensor
        if w_plus:
            assert w.ndim == 3
        else:
            assert w.ndim == 2
            w = w.unsqueeze(1).repeat(1, self.num_ws, 1)
        assert w.shape[1] == self.num_ws
        assert w.shape[2] == self.w_dim
        assert noise_mode in ['none', 'const', 'random']
        if self.network_type == 'eg3d':
            assert c is not None
            image = self.network.synthesis(w, c, noise_mode=noise_mode)
            self.__raw_image = image['image_raw']
            self.__depth_map = image['image_depth']
            return image['image']
        else:
            image = self.network.synthesis(w, noise_mode=noise_mode)
            return image

    def get_depth_map(self) -> torch.Tensor:
        """
            Returns the depth map (for EG3D), the ``synthesis``must
            have been called before
        """
        assert self.network_type == 'eg3d'
        return self.__depth_map

    def get_raw_image(self) -> torch.Tensor:
        """
            Returns the raw image (for EG3D), the ``synthesis``must
            have been called before
        """
        assert self.network_type == 'eg3d'
        return self.__raw_image

    def camera(
            self,
            fov_deg : float = 18.0,
            pitch_deg : float = 0.0,
            yaw_deg : float = 0.0) -> torch.Tensor:
        """
            Returns a camera label (1, 25) from view angles in degrees.
        """
        assert self.network_type == 'eg3d'
        pitch = pi / 2 + pitch_deg * pi / 180
        yaw = pi / 2  + yaw_deg * pi / 180
        cam2world_pose = LookAtPoseSampler.sample(
                                            yaw,
                                            pitch,
                                            torch.tensor([0, 0, 0.2], device=self.device),
                                            radius=2.7,
                                            device=self.device)
        intrinsics = FOV_to_intrinsics(fov_deg, device=self.device)
        return torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

# ---

@click.command(
    help='Generates an image from a random or existing latent')
@click.pass_context
@click.option(
    '--network-type',
    '-nt',
    help='Network',
    type=click.Choice(utils.network_types()),
    default='stylegan2')
@click.option(
    '--network-path',
    '-np',
    help='Path to network weights',
    default=None)
@click.option(
    '--seed',
    '-s',
    type=int,
    help='Seed for z latent generation',
    default=None)
@click.option(
    '--number',
    '-n',
    type=int,
    help='Number of images to generate',
    default=1,
    show_default=True)
@click.option(
    '--input',
    '-i',
    type=str,
    help='Input latent HDF5 file, random z if not specified',
    default=None)
@click.option(
    '--w-plus',
    '-p',
    is_flag=True,
    help='Use w+ if both are available')
@click.option(
    '--trunc',
    'truncation_psi',
    type=float,
    help='Truncation psi',
    default=0.7,
    show_default=True)
@click.option(
    '--noise-mode',
    help='Noise mode',
    type=click.Choice(['const', 'random', 'none']),
    default='const',
    show_default=True)
@click.option(
    '--fov',
    type=float,
    help='Camera FoV in degrees (for EG3D only)',
    default=18.0)
@click.option(
    '--pitch',
    type=float,
    help='Camera pitch in degrees (for EG3D only)',
    default=0.0)
@click.option(
    '--yaw',
    type=float,
    help='Camera FoV in degrees (for EG3D only)',
    default=0.0)
@click.option(
    '--out',
    '-o',
    type=click.Path(file_okay=True, dir_okay=False),
    help='File path to save the results',
    default='out.jpg',
    show_default=True)
@click.option(
    '--save-depth',
    '-d',
    is_flag=True,
    help='Save depth map')
def generate(
        ctx : click.Context,
        network_type : str,
        network_path: str | None,
        seed : int,
        number : int,
        input : str,
        w_plus : bool,
        truncation_psi : float,
        noise_mode : str,
        fov : float,
        pitch : float,
        yaw : float,
        save_depth : bool,
        out: str):
    # setup
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device('cuda')
    generator = Generator(
        network_type=network_type,
        network_path=network_path,
        device=device)
    if network_type == 'eg3d':
        c = generator.camera(fov_deg=fov, pitch_deg=pitch, yaw_deg=yaw)
        if number > 1:
            c = c.expand(number, -1)
    else:
        c = None
    # load or generate latent
    if input is not None:
        number = 1
        sample = Sample()
        sample.load(file_path=input, device=device)
        assert sample.network_type == network_type
        if sample.w_latent is not None and w_plus == False:
            w = sample.w_latent
        elif sample.w_plus_latent is not None:
            w = sample.w_plus_latent
            w_plus = True
    else:
        z = torch.randn((number, generator.z_dim), device=device)
        w = generator.mapping(z, c, truncation_psi=truncation_psi)
    # generate and save image
    image = generator.synthesis(w=w, c=c, w_plus=w_plus, noise_mode=noise_mode)
    if save_depth:
        assert network_type == 'eg3d'
        depth = generator.get_depth_map()
        depth = utils.adjust_dynamic_range(depth)
        depth = einops.repeat(depth, 'b 1 h w -> b c h w', c=3)
        resize = torchvision.transforms.Resize((512,512))
        depth = resize(depth)
        image = torch.cat((image, depth))
        num_rows = number
    else:
        num_rows = 8
    utils.save_image(
            image=image,
            file_path=out,
            create_directories=True,
            num_rows=num_rows)
