#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
#
# heavily adapted from https://github.com/tocantrell/stylegan3-fun-pipeline (DG)
# added FFHQ cropper
# removed most losses
# class structure & full pytorch porting
# added EG3D support
#
# This code was entirely written by a human

import os
import sys
import click
from dataclasses import dataclass
from time import perf_counter

import imageio

import torch
import torch.nn.functional as F
import numpy as np
import einops

from . import utils
from .utils import Sample
from .cropper import Cropper
from .generator import Generator
from .face_extractor_3d import FaceExtractor3D

sys.path.insert(0, os.path.join(utils.source_path, 'synthetics/stylegan3/'))

from synthetics.stylegan3.metrics import metric_utils
from synthetics.stylegan3 import dnnlib
from synthetics.stylegan3.torch_utils import gen_utils

# ---

class Projector():
    """ Define a optimizing projector on the W or W+ latent space. """

    @dataclass
    class ProjectorParameters:
        num_steps : int = 1000
        w_avg_samples : int = 10000
        projection_seed : int = None
        project_in_wplus : bool = False
        start_wavg : bool = True
        truncation_psi : float = 0.7
        initial_learning_rate : float = 0.1
        initial_noise_factor : float = 0.05
        constant_learning_rate : bool = False
        lr_rampdown_length : float = 0.25
        lr_rampup_length : float = 0.05
        noise_ramp_length: float = 0.75
        regularize_noise_weight : float = 1e5

    # ---

    def __init__(
            self,
            network_type : str = 'stylegan3',
            loss_type : str = 'sgan2',
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32,
            parameters : ProjectorParameters = ProjectorParameters(),
            ) -> None:
        self.network_type = network_type
        assert loss_type in ['sgan2']
        self.loss_type = loss_type
        self.device = device
        self.dtype  = dtype
        self.generator = Generator(
            network_type=self.network_type,
            device=self.device,
            dtype=self.dtype)
        self.w_dim = self.generator.w_dim
        self.z_dim = self.generator.z_dim
        self.img_resolution = self.generator.img_resolution
        self.num_ws = self.generator.num_ws
        self.w_avg = self.generator.w_avg
        self.parameters = parameters

    # ---

    def get_config(self) -> dict:
        return \
        {
            'optimization_options':
            {
                'num_steps': self.parameters.num_steps,
                'initial_learning_rate': self.parameters.initial_learning_rate,
                'constant_learning_rate': self.parameters.constant_learning_rate,
                'regularize_noise_weight': self.parameters.regularize_noise_weight,
            },
            'projection_options':
            {
                'w_avg_samples': self.parameters.w_avg_samples,
                'initial_noise_factor': self.parameters.initial_noise_factor,
                'lr_rampdown_length': self.parameters.lr_rampdown_length,
                'lr_rampup_length': self.parameters.lr_rampup_length,
                'noise_ramp_length': self.parameters.noise_ramp_length,
            },
            'latent_space_options':
            {
                'project_in_wplus': self.parameters.project_in_wplus,
                'start_wavg': self.parameters.start_wavg,
                'projection_seed': self.parameters.projection_seed,
                'truncation_psi': self.parameters.truncation_psi,
            },
            'loss_options':
            {
                'loss_type': self.loss_type,
            }
        }

    # ---

    def get_start_vector(
            self,
            target_pose : torch.Tensor | None,
            start_vector : torch.Tensor | None,
            ) -> tuple[torch.Tensor]:
        """
            Get the start vector.
        """
        # Check or generate c label for EG3D
        if target_pose is not None:
            assert target_pose.shape == (1, 25)
            c_label = target_pose
        elif self.network_type == 'eg3d':
            c_label = self.generator.camera()
        else:
            c_label = None

        # Sample z vectors
        z_samples = torch.randn(
            (self.parameters.w_avg_samples, self.z_dim),
            device=self.device,
            dtype=self.dtype)

        # Setup camera label
        if self.network_type == 'eg3d':
            c_samples = c_label.repeat(z_samples.shape[0], 1)
        else:
            c_samples = None

        # Compute w stats
        click.echo(f'Generating W stats using {self.parameters.w_avg_samples} samples...')
        w_samples = self.generator.mapping(z_samples, c_samples)
        w_avg = torch.mean(w_samples, dim=0, keepdim=True)
        w_std = (torch.sum((w_samples - w_avg) ** 2) / self.parameters.w_avg_samples) ** 0.5

        # Generate start vector
        if start_vector is not None:
            click.echo(f'Starting from provided vector...')
            w_start = start_vector
        elif self.parameters.start_wavg:
            click.echo(f'Starting from W midpoint...')
            w_start = w_avg
        else:
            click.echo(f'Starting from a random vector (seed: {self.parameters.projection_seed})...')
            z = torch.randn(
                (1, self.z_dim),
                device=self.device,
                dtype=self.dtype)
            w_start = self.generator.mapping(
                z=z,
                c=c_label,
                truncation_psi=self.parameters.truncation_psi)

        # Extend vectors for W+ projections
        if self.parameters.project_in_wplus:
            click.echo('Projecting in W+ latent space...')
            w_start = w_start.repeat([1, self.num_ws, 1])
        else:
            click.echo('Projecting in W latent space...')

        return w_start, w_std, c_label

    def project(
            self,
            target: torch.Tensor,
            target_depth_map : torch.Tensor | None = None,
            target_pose : torch.Tensor | None = None,
            start_vector : torch.Tensor | None = None
            ) -> torch.Tensor:
        """
            Projecting a 'target' image into the W latent space. The user has
            an option to project into W+, where all elements in the latent
            vector are different. Likewise, the projection process can start
            from the W midpoint or from a random point, though results have
            shown that starting from the midpoint (start_wavg) yields the
            best results.

            target: [B,C,H,W] and dynamic range [-1.0,1.0],
            W & H must match the generator output resolution.

            returns: torch tensor, output shape: [num_steps, 1, num_ws, 512]
        """
        assert target.ndim == 4
        assert target.shape[0] == 1, 'projection does not support batch mode yet'
        assert target.shape[1] == 3
        # Consider resizing to network input resolution
        #target = torch.nn.functional.interpolate(target, size=(self.img_resolution, self.img_resolution))
        assert target.shape[2] == self.img_resolution
        assert target.shape[3] == self.img_resolution
        if target_depth_map is not None:
            assert target.ndim == 4
            assert target_depth_map.shape[0] == 1
            assert target_depth_map.shape[1] == 1
            #assert target_depth_map.shape[2] == self.i_res
            #assert target_depth_map.shape[3] == self.i_res
        batch_size = target.shape[0]

        # Seed
        if self.parameters.projection_seed is not None:
            torch.manual_seed(self.parameters.projection_seed)

        # Get start vector
        w_start, w_std, c_label = self.get_start_vector(
            target_pose=target_pose,
            start_vector=start_vector)

        # Setup noise inputs (only for StyleGAN2 models)
        if self.network_type == 'stylegan2':
            snbs = self.generator.network.synthesis.named_buffers()
            noise_buffs = {name: buf for (name, buf) in snbs if 'noise_const' in name}
        else:
            noise_buffs = None

        # Features for target image. Reshape to 256x256 if it's larger to use with VGG16
        if self.loss_type in ['sgan2']:
            target = (target + 1.0) * (255.0/2.0)
            if target.shape[2] > 256:
                target = F.interpolate(target, size=(256, 256), mode='area')
            vgg16 = metric_utils.get_feature_detector(
                utils.get_model_path('vgg16'),
                device=self.device)
            target_features = vgg16(target, resize_images=False, return_lpips=True)

        # Copy start vector
        w_opt = w_start.clone().detach().requires_grad_(True)

        # Allocate output vectors
        w_out = torch.empty(
            (self.parameters.num_steps, batch_size, self.num_ws, self.generator.w_dim),
            dtype=self.dtype,
            device=self.device)

        # Init optimizer
        if self.network_type == 'stylegan2':
            optimizer = torch.optim.Adam(
                [w_opt] + list(noise_buffs.values()),
                betas=(0.9, 0.999),
                lr=self.parameters.initial_learning_rate)
        else:
            optimizer = torch.optim.Adam(
                [w_opt], betas=(0.9, 0.999),
                lr=self.parameters.initial_learning_rate)

        # Init noise
        if self.network_type == 'stylegan2':
            for buf in noise_buffs.values():
                buf[:] = torch.randn_like(buf)
                buf.requires_grad = True

        # Optimization loop
        for step in range(self.parameters.num_steps):
            # Learning rate schedule.
            t = step / self.parameters.num_steps
            w_noise_scale = w_std * self.parameters.initial_noise_factor \
                            * max(0.0, 1.0 - t / self.parameters.noise_ramp_length) ** 2

            if self.parameters.constant_learning_rate:
                # Turn off the rampup/rampdown of the learning rate
                lr_ramp = 1.0
            else:
                lr_ramp = min(1.0, (1.0 - t) / self.parameters.lr_rampdown_length)
                lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp = lr_ramp * min(1.0, t / self.parameters.lr_rampup_length)
            lr = self.parameters.initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            if self.parameters.project_in_wplus:
                ws = w_opt + w_noise
            else:
                ws = (w_opt + w_noise).repeat([1, self.num_ws, 1])
            synth_images = self.generator.synthesis(
                w=ws,
                c=c_label,
                w_plus=True,
                noise_mode='const')

            # Down-sample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255/2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            if self.loss_type == 'sgan2':
                synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
                dist = (target_features - synth_features).square().sum()

                # Noise regularization.
                reg_loss = 0.0
                if self.network_type == 'stylegan2':
                    for v in noise_buffs.values():
                        noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                        while True:
                            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                            if noise.shape[2] <= 8:
                                break
                            noise = F.avg_pool2d(noise, kernel_size=2)
                    loss = dist + reg_loss * self.parameters.regularize_noise_weight
                else:
                    loss = dist
                # Print in the same line (avoid cluttering the command line)
                n_digits = int(np.log10(self.parameters.num_steps)) + 1 if self.parameters.num_steps > 0 else 1
                message = f'step {step + 1:{n_digits}d}/{self.parameters.num_steps}: dist {dist:.7e} | loss {loss.item():.7e}'
                print(message, end='\r')

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Save projected W for each optimization step.
            if self.parameters.project_in_wplus:
                w_out[step] = w_opt.detach()
            else:
                w_out[step] = w_opt.repeat([1, self.num_ws, 1]).detach()

            # Normalize noise.
            if self.network_type == 'stylegan2':
                with torch.no_grad():
                    for buf in noise_buffs.values():
                        buf -= buf.mean()
                        buf *= buf.square().mean().rsqrt()

        # Return vectors
        return w_out

# ---

@click.command(
    help='Project a given image to the latent space')
@click.pass_context
@click.option(
    '--network-type',
    '-nt',
    help='Network type',
    type=click.Choice(utils.network_types()),
    default='stylegan2')
@click.option(
    '--target',
    '-t',
    'target_fname',
    type=click.Path(exists=True, dir_okay=False),
    help='Target image file to project to',
    required=True,
    metavar='FILE')
@click.option(
    '--crop',
    '-c',
    is_flag=True,
    help='Use face cropper for preprocessing')
@click.option(
    '--num-steps',
    '-n',
    help='Number of optimization steps',
    type=click.IntRange(min=0),
    default=1000,
    show_default=True)
@click.option(
    '--init-lr',
    '-lr',
    'initial_learning_rate',
    type=float,
    help='Initial learning rate of the optimization process',
    default=0.1,
    show_default=True)
@click.option(
    '--constant-lr',
    'constant_learning_rate',
    is_flag=True,
    help=   'Add flag to use a constant learning rate throughout the '
            'optimization (turn off the rampup/rampdown)')
@click.option(
    '--reg-noise-weight',
    '-regw',
    'regularize_noise_weight',
    type=float,
    help='Noise weight regularization',
    default=1e5,
    show_default=True)
@click.option(
    '--stabilize-projection',
    is_flag=True,
    help=   'Add flag to stabilize the latent space/anchor to w_avg, '
            'making it easier to project (only for StyleGAN3 config-r/t models)')
@click.option(
    '--save-video',
    '-v',
    is_flag=True,
    help='Save an mp4 video of optimization progress')
@click.option(
    '--compress',
    is_flag=True,
    help='Compress video with ffmpeg-python; same resolution, lower memory size')
@click.option(
    '--fps',
    type=int,
    help='FPS for the mp4 video of optimization progress (if saved)',
    default=30,
    show_default=True)
@click.option(
    '--project-in-wplus',
    '-p',
    is_flag=True,
    help='Project in the W+ latent space')
@click.option(
    '--start-wavg',
    '-a',
    type=bool,
    help=   'Start with the average W vector, otherwise will start '
            'from a random seed (provided by user)',
    default=True,
    show_default=True)
@click.option(
    '--projection-seed',
    '-s',
    type=int,
    help='Seed to start projection from',
    default=123,
    show_default=True)
@click.option(
    '--trunc',
    'truncation_psi',
    type=float,
    help='Truncation psi to use in projection when using a projection seed',
    default=0.7,
    show_default=True)
@click.option(
    '--outdir',
    '-o',
    type=click.Path(file_okay=False),
    help='Directory path to save the results',
    default=os.path.join(os.getcwd(), 'out', 'projection'),
    show_default=True,
    metavar='DIR')
@click.option(
    '--description',
    '-desc',
    type=str,
    help='Extra description to add to the experiment name',
    default='')
@click.option(
    '--device',
    '-d',
    type=click.Choice(
        ['cpu', 'cuda:0'],
        case_sensitive=False),
    help='Device',
    default='cuda:0')
def project(
        ctx: click.Context,
        network_type: str,
        target_fname: str,
        crop: bool,
        num_steps: int,
        initial_learning_rate: float,
        constant_learning_rate: bool,
        regularize_noise_weight: float,
        stabilize_projection: bool,
        save_video: bool,
        compress: bool,
        fps: int,
        project_in_wplus: bool,
        start_wavg: bool,
        projection_seed: int,
        truncation_psi: float,
        outdir: str,
        description: str,
        device : str):
    # Init projector
    device = torch.device(device)
    parameters = Projector.ProjectorParameters(
        num_steps=num_steps,
        initial_learning_rate=initial_learning_rate,
        constant_learning_rate=constant_learning_rate,
        regularize_noise_weight=regularize_noise_weight,
        project_in_wplus=project_in_wplus,
        start_wavg=start_wavg,
        truncation_psi=truncation_psi,
        projection_seed=projection_seed
    )
    projector = Projector(
        network_type=network_type,
        device=device,
        dtype=torch.float32,
        parameters=parameters)

    # Load target and crop
    target = utils.load_image(target_fname, device=device)
    if crop:
        input_config = Cropper.Config.dlib
        if network_type == 'stylegan2' or network_type == 'stylegan3':
            cropper = Cropper(
                input_config=input_config,
                output_config=Cropper.Config.ffhq,
                device=device)
        elif network_type == 'stylegan2-256':
            cropper = Cropper(
                input_config=input_config,
                output_config=Cropper.Config.ffhq256,
                device=device)
        elif network_type == 'eg3d':
            cropper = Cropper(
                input_config=input_config,
                output_config=Cropper.Config.eg3d,
                device=device)
        else:
            raise RuntimeError('Unknown network type')
        target = cropper.crop(target)

    # Get 3D pose, camera information and depth map (for EG3D)
    if network_type == 'eg3d':
        face_extractor = FaceExtractor3D(
            device=device)
        face_extractor_cropper = Cropper(
            input_config=Cropper.Config.dlib,
            output_config=Cropper.Config.deep3dfr,
            device=device)
        target_3dfr = face_extractor_cropper.crop(target)
        face_extractor.extract(image=target_3dfr)
        target_depth_map = face_extractor.get_depth_map()
        target_pose = face_extractor.compute_pose()['label']
    else:
        target_depth_map = None
        target_pose = None

    # Stabilize the latent space to make things easier (for StyleGAN3's config t and r models)
    if stabilize_projection:
        gen_utils.anchor_latent_space(projector.network)

    # Optimize projection.
    start_t = perf_counter()
    projected_w_steps = projector.project(
        target=target,
        target_depth_map=target_depth_map,
        target_pose=target_pose,
        )
    elapsed_time = dnnlib.util.format_time(perf_counter()-start_t)
    run_config = projector.get_config()
    print(f'\nElapsed time: {elapsed_time}')
    run_config['elapsed_time'] = elapsed_time
    # Make the run dir automatically
    desc = 'projection-wplus' if project_in_wplus else 'projection-w'
    desc = f'{desc}-wavgstart' if start_wavg else f'{desc}-seed{projection_seed}start'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    run_dir = gen_utils.make_run_dir(outdir, desc)

    # Save the configuration used
    ctx.obj = \
    {
        'network_type': network_type,
        'description': description,
        'target_image': target_fname,
        'outdir': run_dir,
        'save_video': save_video,
        'seed': projection_seed,
        'video_fps': fps,
        'run_config': run_config
    }

    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    # Render debug output: optional video and projected image and W vector.
    result_name = os.path.join(run_dir, 'proj')
    sample_name = os.path.join(run_dir, 'projected')

    # If we project in W+, add to the name of the results
    if project_in_wplus:
        result_name, sample_name = f'{result_name}_wplus', f'{sample_name}_wplus'

    # Either in W or W+, we can start from the W midpoint or one given by the projection seed
    if start_wavg:
        result_name, sample_name = f'{result_name}_wavg', f'{sample_name}_wavg'
    else:
        result_name, sample_name = f'{result_name}_seed-{projection_seed}', f'{sample_name}_seed-{projection_seed}'

    # Save the target image
    print('Saving projection results...')
    utils.save_image(target, os.path.join(run_dir, 'target.jpg'))
    w = projected_w_steps[-1]
    synth_image = projector.generator.synthesis(
        w=w,
        c=target_pose,
        w_plus=True,
        noise_mode='const')
    utils.save_image(synth_image, f'{result_name}_final.jpg')
    if projector.network_type == 'eg3d':
        depth_image = projector.generator.get_depth_map()
        utils.save_image(depth_image, f'{result_name}_depth.jpg', normalize=True)

    # Save the latent vector
    if project_in_wplus:
        sample = Sample(
            w_plus_latent=w,
            c_label=target_pose,
            network_type=network_type)
    else:
        sample = Sample(
            w_latent=w[:,0,:],
            c_label=target_pose,
            network_type=network_type)
    sample.save(file_path=f'{sample_name}_final.h5')

    # Save the optimization video and compress it if so desired
    if save_video:
        video = imageio.get_writer(
            f'{result_name}.mp4',
            mode='I',
            fps=fps,
            codec='libx264',
            bitrate='16M')
        print(f'Saving optimization progress video "{result_name}.mp4"')
        target_uint8 = utils.image_to_numpy(target)[0]
        target_uint8 = einops.rearrange(target_uint8, 'c h w -> h w c')
        for projected_w in projected_w_steps:
            synth_image = projector.generator.synthesis(
                w=projected_w,
                c=target_pose,
                w_plus=True,
                noise_mode='const')
            synth_image = utils.image_to_numpy(synth_image)
            synth_image = synth_image[0]
            synth_image = einops.rearrange(synth_image, 'c h w -> h w c')
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Compress the video; might fail, and is a basic command that can also be better optimized
    if save_video and compress:
        gen_utils.compress_video(original_video=f'{result_name}.mp4',
                                 original_video_name=f'{result_name.split(os.sep)[-1]}',
                                 outdir=run_dir,
                                 ctx=ctx)
