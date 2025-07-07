#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import os
import math
import glob
import time
from dataclasses import dataclass

import click
from tqdm import tqdm
import h5py
import yaml

import torch
import einops
import numpy as np

from . import utils
from .utils import Sample, SampleCollection
from .latent_edit import LatentEdit
from .generator import Generator
from .cropper import Cropper
from .embedding import Embedding

# ---

class DatabaseGenerator:
    """ Generate a synthetic images database. """

    # ---

    @dataclass
    class ParallelParameters:
        """ Default batch parameters """
        batch_size : int = 8
        job_parallel : bool = False
        job_rank : int = 0
        job_num : int = 0

    # ---

    REFERENCES_ALGORITHMS = ['random', 'reject', 'repulse', 'langevin']

    # ---

    @dataclass
    class ReferencesParameters:
        ict : float = 1.0
        repulse : float = 0.0
        max_iterations_repulse : int = 10
        neutralisation : bool = False
        rescale_latents : bool = False
        camera_fov : float = 20

    # ---

    @dataclass
    class LangevinParameters:
        """ Default Langevin parameters """
        iterations : int = 10
        checkpoint_iter : int = -1
        repulsion_coefficient : float = 1.0
        repulsion_radius : float = 1.7
        timestep : float = -0.3
        viscous_coefficient : float = 1.0
        random_force : float = 0.01
        latent_force : float = 0.0
        constant_embeddings_path : str = None
        constant_distance : float = 0.0

    # ---

    VARIATIONS_ALGORITHMS = ['dispersion', 'covariates']

    # ---

    @dataclass
    class DispersionParameters:
        """ Default Dispersion parameters """
        iterations : int = 20
        checkpoint_iter : int = -1
        num_augmentations : int = 16
        embedding_coefficient : float = 20.0
        latent_coefficient : float = 1.0
        latent_radius : float = 12.0
        latent_global_coefficient : float = 1.0
        random_force : float = 0.05
        viscous_coefficient : float = 1.0
        timestep : float = 0.05
        initial_noise : float = 0.2
        initial_covariates : bool = False
        camera_fov : float = 20
        camera_angle_variation : float = -1.0

    # ---

    def __init__(
            self,
            identities : list[int],
            network_type : str = 'stylegan2',
            network_path: str | None = None,
            embedding_type : str = 'iresnet50',
            postprocessor_config : str | None = None,
            cropper_config : Cropper.Config | None = None,
            truncation_psi : float = 0.7,
            root_directory : str = None,
            create_directories : bool = True,
            parallel_parameters : ParallelParameters = ParallelParameters(),
            disable_progress_bar : bool = False,
            generate_images : bool = True,
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32,
            seed : int = None):
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.identities = identities
        self.create_directories = create_directories
        self.disable_progress_bar = disable_progress_bar
        self.generate_images = generate_images
        self.network_type = network_type
        self.embedding_type = embedding_type
        self.truncation_psi = truncation_psi
        self.latent_edit = None
        self.parallel_parameters = parallel_parameters
        self.generator = Generator(
            require_grad=True,
            network_type=self.network_type,
            network_path=network_path,
            device=device,
            dtype=self.dtype)
        self.embedding = Embedding(
            model_name=self.embedding_type,
            device=self.device,
            dtype=self.dtype)
        if cropper_config is None:
            if self.network_type == 'eg3d':
                cropper_input_config = Cropper.Config.mtcnn
            elif self.network_type == 'stylegan3-casia-128':
                cropper_input_config = Cropper.Config.resize
            elif self.network_type in ['stylegan2-256', 'stylegan3-t-265', 'stylegan3-r-265']:
                cropper_input_config = Cropper.Config.ffhq256
            elif "lucidrains" in self.network_type:
                size = self.network_type.split("-")[-1]
                cropper_key = f"ffhq{size}"
                cropper_input_config = Cropper.Config[cropper_key]
            else:
                cropper_input_config = Cropper.Config.ffhq
        else:
            cropper_input_config = cropper_config
        cropper_output_config = self.embedding.cropper_config
        self.cropper = Cropper(
            input_config=cropper_input_config,
            output_config=cropper_output_config,
            device=self.device,
            dtype=self.dtype)
        assert root_directory is not None, 'A root directory must be specified'
        if not os.path.exists(root_directory):
            if create_directories:
                os.makedirs(root_directory)
            else:
                raise RuntimeError('Root directory does not exist')
        self.root_directory = root_directory
        if postprocessor_config is not None:
            self.postprocessor = Cropper(
                input_config=cropper_input_config,
                output_config=postprocessor_config,
                device=self.device,
                dtype=self.dtype)
            postprocessor_config_item = Cropper.get_config_item(postprocessor_config)
            w = postprocessor_config_item.width
            h = postprocessor_config_item.height
            if w is None or h is None:
                image_directory_name = f'images_{postprocessor_config}'
            else:
                image_directory_name = f'images_{postprocessor_config}_{w}x{h}'
        else:
            self.postprocessor = None
            image_directory_name = f'images'
        self.image_directory = os.path.join(root_directory, image_directory_name)
        samples_collection_name = 'samples.h5'
        samples_collection_path = os.path.join(
            self.root_directory, samples_collection_name)
        self.sample_collection = SampleCollection(
            file_path=samples_collection_path,
            read_only=False)
        # Load existing samples if file exists
        if os.path.exists(samples_collection_path):
            self.sample_collection.load(
                device=self.device,
                dtype=self.dtype,
                identities=self.identities)

    # ---

    def get_image_path(
            self,
            identity : int,
            label : str,
            image_directory : str | None = None
            ) -> str:
        """ Return the path to an image given identity and label. """
        if image_directory is None:
            image_directory = self.image_directory
        file_path = os.path.join(image_directory, f'{identity:05}', f'{label}.png')
        return file_path

    # ---

    def save_images(
            self,
            identities : list[int],
            latents : torch.Tensor,
            cameras : torch.Tensor | None,
            label: str | list[str] = 'reference',
            image_directory : str | None = None
            ) -> None:
        """ Save images from latents. Latent array can be either 2d or 3d. """
        assert isinstance(identities, list)
        num_identities = len(identities)
        assert isinstance(latents, torch.Tensor)
        if latents.ndim == 2:
            if latents.shape[0] == num_identities:
                pass
            elif latents.shape[0] >=num_identities:
                latents = latents[0:num_identities, :]
            else:
                raise RuntimeError('Tensor dimension (d=0) smaller than number of identities')
            assert isinstance(label, str)
            samples_per_id = None
        elif latents.ndim == 3:
            if latents.shape[0] == num_identities:
                pass
            elif latents.shape[0] >=num_identities:
                latents = latents[0:num_identities, :, :]
            else:
                raise RuntimeError('Tensor dimension (d=0) smaller than number of identities')
            assert isinstance(label, list)
            samples_per_id = len(label)
            if latents.shape[1] == samples_per_id:
                pass
            elif latents.shape[1] >= len(label):
                latents = latents[:, 0:samples_per_id, :]
            else:
                raise RuntimeError('Tensor dimension (d=1) smaller than number of labels')
        else:
            raise Exception('Unsupported tensor dimension')
        if cameras is not None:
            assert cameras.ndim == latents.ndim
            assert cameras.shape[0] == latents.shape[0]
            if cameras.ndim == 3:
                assert cameras.shape[1] == latents.shape[1]
                assert cameras.shape[2] == 25
            else:
                assert cameras.shape[1] == 25
        if image_directory is None:
            image_directory = self.image_directory
        if not os.path.exists(image_directory):
            if self.create_directories:
                os.makedirs(image_directory)
            else:
                raise RuntimeError('Image directory does not exist')
        latent_batch = torch.empty(
            size=(self.parallel_parameters.batch_size, self.generator.w_dim),
            device=self.device,
            dtype=self.dtype)
        if cameras is not None:
            camera_batch = torch.empty(
                size=(self.parallel_parameters.batch_size, 25),
                device=self.device,
                dtype=self.dtype)
        else:
            camera_batch = None
        path_batch : list[str] = []
        def finish_image_batch() -> None:
            nonlocal path_batch
            img_batch = self.generate_image_from_w(w=latent_batch, c=camera_batch)
            img_post_batch = self.postprocess_image(img=img_batch)
            for i, path in enumerate(path_batch):
                utils.save_image(
                    image=img_post_batch[i].unsqueeze(0),
                    file_path=path,
                    create_directories=True)
            path_batch = []
        def save_image_batch(
                latent : torch.Tensor,
                camera : torch.Tensor | None,
                path : str) -> list[str]:
            nonlocal latent_batch
            nonlocal camera_batch
            nonlocal path_batch
            i = len(path_batch)
            latent_batch[i, :] = latent
            if camera is not None:
                camera_batch[i, :] = camera
            assert isinstance(path, str)
            path_batch.append(path)
            if i + 1 == self.parallel_parameters.batch_size:
                finish_image_batch()
        if samples_per_id is None: # 2D case
            for i, identity in enumerate(
                    tqdm(identities, disable=self.disable_progress_bar)):
                file_path=self.get_image_path(
                    identity=identity,
                    label=label,
                    image_directory=image_directory)
                save_image_batch(
                    latent=latents[i],
                    camera=cameras[i] if cameras is not None else None,
                    path=file_path)
            finish_image_batch()
        else: # 3D case
            for i, identity in enumerate(
                    tqdm(identities, disable=self.disable_progress_bar)):
                for a in range(samples_per_id):
                    file_path=self.get_image_path(
                        identity=identity,
                        label=label[a],
                        image_directory=image_directory)
                    save_image_batch(
                        latent=latents[i, a],
                        camera=cameras[i, a] if cameras is not None else None,
                        path=file_path)
                finish_image_batch()

    # ---

    def generate_all_images(
            self,
            image_directory: str | None = None,
            identity_batch_size: int = 1024) -> None:
        """ Generate images from the samples.h5 collection. """
        assert not self.parallel_parameters.job_parallel
        click.echo('Generating images...')
        if image_directory is None:
            image_directory = self.image_directory
        num_identities = len(self.identities)
        labels_ref_0 = self.sample_collection.list_identity_labels(
            identity=self.identities[0])
        num_labels_per_identity = []
        all_labels_similar = True
        for identity in self.identities:
            identity_labels = self.sample_collection.list_identity_labels(
                identity=identity)
            num_identity_labels = len(identity_labels)
            num_labels_per_identity.append(num_identity_labels)
            if all_labels_similar:
                if identity_labels != labels_ref_0:
                    all_labels_similar = False
        latent_buffer = torch.empty(
            size=(num_identities, max(num_labels_per_identity), self.generator.w_dim),
            device=self.device,
            dtype=self.dtype)
        if self.network_type == 'eg3d':
            camera_buffer = torch.empty(
                size=(num_identities, max(num_labels_per_identity), 25),
                device=self.device,
                dtype=self.dtype)
        else:
            camera_buffer = None
        if not all_labels_similar:
            identity_batch_size = 1
        for idx in tqdm(
                range(0, num_identities, identity_batch_size),
                disable=self.disable_progress_bar):
            idx_start = idx
            idx_stop = min(idx + identity_batch_size, num_identities)
            idx_num = idx_stop - idx_start
            identity_batch = self.identities[idx_start:idx_stop]
            for i, identity in enumerate(identity_batch):
                if all_labels_similar:
                    labels = labels_ref_0
                else:
                    labels = self.sample_collection.list_identity_labels(
                        identity=identity)
                for a, label in enumerate(labels):
                    sample = self.sample_collection.get_sample(
                        identity=identity,
                        label=label)
                    latent_buffer[i, a, :] = sample.w_latent[0, :]
                    if camera_buffer is not None:
                        camera_buffer[i, a, :] = sample.c_label[0, :]
            self.save_images(
                identities=identity_batch,
                label=labels,
                latents=latent_buffer[0:idx_num, :, :],
                cameras=camera_buffer[0:idx_num, :, :] if camera_buffer is not None else None,
                image_directory=image_directory)

    # ---

    def postprocess_image(
            self,
            img : torch.Tensor,
            detach : bool = True,
            echo_batch : bool = False
            ) -> torch.Tensor:
        """ Postprocess an image (batch). """
        num_images = img.shape[0]
        if self.postprocessor is None:
            if detach:
                return img.clone().detach()
            else:
                return img
        if num_images <= self.parallel_parameters.batch_size:
            img_post = self.postprocessor.crop(img)
        else:
            img_post = torch.empty(
                (num_images,
                    3,
                    self.postprocessor.output_height,
                    self.postprocessor.output_width),
                device=self.device,
                dtype=self.dtype)
            for i in range(0, num_images, self.parallel_parameters.batch_size):
                if echo_batch:
                    click.echo(f'... batch {i}/{num_images}')
                num_items = min(self.parallel_parameters.batch_size, num_images - i)
                img_chunk = img[i:i+num_items]
                img_post_chunk = self.postprocessor.crop(image=img_chunk)
                if detach:
                    img_post_chunk = img_post_chunk.detach()
                img_post[i:i+num_items] = img_post_chunk
        if detach:
            img_post = img_post.detach()
        return img_post

    # ---

    def save_to_collection(
            self,
            collection : SampleCollection,
            identities : list[int],
            latents : torch.Tensor,
            z_latents : torch.Tensor | None = None,
            embeddings : torch.Tensor | None = None,
            camera: torch.Tensor | None = None,
            label: str | list[str] = 'reference'
            ) -> None:
        """ Save to sample collection helper. Arrays can be either 2d or 3d. """
        assert isinstance(collection, SampleCollection)
        assert not collection.read_only
        if latents.ndim == 2:
            samples_per_id = None
            assert isinstance(label, str)
        elif latents.ndim == 3:
            samples_per_id = latents.shape[1]
            assert isinstance(label, list)
            assert len(label) == samples_per_id
        else:
            raise Exception('Unsupported tensor dimension')
        assert latents.shape[0] == len(identities)
        if z_latents is not None:
            assert z_latents.ndim == latents.ndim
            assert z_latents.shape[0] == len(identities)
        if embeddings is not None:
            assert embeddings.ndim == latents.ndim
            assert embeddings.shape[0] == len(identities)
        if camera is not None:
            assert camera.ndim == latents.ndim
            assert camera.shape[0] == latents.shape[0]
            if camera.ndim == 3:
                assert camera.shape[1] == latents.shape[1]
                assert camera.shape[2] == 25
            else:
                assert camera.shape[1] == 25
        if samples_per_id is None: # 2D case
            for i, identity in enumerate(
                    tqdm(identities, disable=self.disable_progress_bar)):
                w = latents[i].unsqueeze(0)
                z = z_latents[i].unsqueeze(0) if z_latents is not None else None
                e = embeddings[i].unsqueeze(0) if embeddings is not None else None
                c = camera[i].unsqueeze(0) if camera is not None else None
                sample = Sample(
                    w_latent=w,
                    z_latent=z,
                    c_label=c,
                    embedding=e,
                    network_type=self.network_type,
                    embedding_type=self.embedding_type)
                collection.add_sample(
                    identity=identity,
                    label=label,
                    sample=sample)
        else: # 3D case
            for i, identity in enumerate(
                    tqdm(identities, disable=self.disable_progress_bar)):
                for a in range(samples_per_id):
                    w = latents[i, a].unsqueeze(0)
                    z = z_latents[i, a].unsqueeze(0) if z_latents is not None else None
                    e = embeddings[i, a].unsqueeze(0) if embeddings is not None else None
                    c = camera[i, a].unsqueeze(0) if camera is not None else None
                    sample = Sample(
                        w_latent=w,
                        z_latent=z,
                        c_label=c,
                        embedding=e,
                        network_type=self.network_type,
                        embedding_type=self.embedding_type)
                    collection.add_sample(
                        identity=identity,
                        label=label[a],
                        sample=sample)
        collection.save()

    # ---

    def merge_sample_collections(
            self,
            reference_collection : SampleCollection | None = None,
            remove : bool = False
            ) -> None:
        """ Merge sample files in root directory. """

        device = torch.device('cpu')
        click.echo('Merging sample collections...')
        collections_paths = glob.glob(
            os.path.join(self.root_directory, 'samples_*.h5'))
        click.echo(f'Found {len(collections_paths)} files, merging ...')
        output_collection_path = os.path.join(self.root_directory, 'samples.h5')
        output_collection = None
        if reference_collection is not None:
            if reference_collection.file_path == output_collection_path:
                if reference_collection.read_only:
                    raise RuntimeError('Output collection is read only')
                else:
                    output_collection = reference_collection
                    copy_references = False
            else:
                copy_references = True
        if output_collection is None:
            output_collection = SampleCollection(output_collection_path)
        if copy_references:
            reference_collection.load(device=device)
            for identity in reference_collection.list_identities():
                label = 'reference'
                sample = reference_collection.get_sample(
                    identity=identity,
                    label=label)
                output_collection.add_sample(
                    identity=identity,
                    label=label,
                    sample=sample)
        for collection_path in collections_paths:
            click.echo(f'-> {collection_path}')
            collection = SampleCollection(
                file_path=collection_path,
                read_only=True)
            collection.load(device=device)
            for identity in collection.list_identities():
                for label in collection.list_identity_labels(identity):
                    sample = collection.get_sample(
                        identity=identity,
                        label=label)
                    output_collection.add_sample(
                        identity=identity,
                        label=label,
                        sample=sample)
        output_collection.save()
        if remove:
            for collection_path in collections_paths:
                os.remove(collection_path)

    # ---

    def torch_seed(
            self,
            add_to_seed : int = 0
            ) -> None:
        """ Seed the random number generator. """
        if self.seed is not None:
            torch.manual_seed(self.seed + add_to_seed)

    # ---

    def generate_z_latents(
            self,
            num_latents : int
            ) -> torch.Tensor:
        """ Generate Z space latent vectors. """
        z = torch.randn(
            num_latents,
            self.generator.z_dim,
            device=self.device,
            dtype=self.dtype)
        return z

    # ---

    def map_z_to_w(
            self,
            z : torch.Tensor,
            c : torch.Tensor | None
            ) -> torch.Tensor:
        """ Map a Z latent to a W latent. """
        if c is not None:
            assert c.ndim == 2
            assert c.shape[0] == z.shape[0]
            assert c.shape[1] == 25
        w = self.generator.mapping(
            z=z,
            c=c,
            truncation_psi=self.truncation_psi)
        return w

    # ---

    def generate_image_from_w(
            self,
            w : torch.Tensor,
            c : torch.Tensor | None,
            detach : bool = True,
            echo_batch : bool = False
            ) -> torch.Tensor:
        """ Generate an image (batch) from a W vector (batch). """
        num_images = w.shape[0]
        if c is not None:
            assert c.ndim == 2
            assert c.shape[0] == num_images
            assert c.shape[1] == 25
        if num_images <= self.parallel_parameters.batch_size:
            img = self.generator.synthesis(
                w=w,
                c=c)
        else:
            img = torch.empty(
                (num_images,
                    3,
                    self.generator.img_resolution,
                    self.generator.img_resolution),
                device=self.device,
                dtype=self.dtype)
            for i in range(0, num_images, self.parallel_parameters.batch_size):
                if echo_batch:
                    click.echo(f'... batch {i}/{num_images}')
                num_vectors = min(self.parallel_parameters.batch_size, num_images - i)
                w_chunk = w[i:i+num_vectors]
                c_chunk = c[i:i+num_vectors] if c is not None else None
                img_chunk = self.generator.synthesis(w=w_chunk, c=c_chunk)
                if detach:
                    img_chunk = img_chunk.detach()
                img[i:i+num_vectors] = img_chunk
        if detach:
            img = img.detach()
        return img

    # ---

    def extract_embedding_from_image(
            self,
            img : torch.Tensor,
            detach : bool = True,
            echo_batch : bool = False
            ) -> torch.Tensor:
        """ Extract embedding from an image (batch). """
        num_images = img.shape[0]
        if num_images <= self.parallel_parameters.batch_size:
            img_crop = self.cropper.crop(image=img)
            e = self.embedding.extract(image=img_crop)
        else:
            e = torch.empty(
                (num_images, self.embedding.e_dim),
                device=self.device,
                dtype=self.dtype)
            for i in range(0, num_images, self.parallel_parameters.batch_size):
                if echo_batch:
                    click.echo(f'... batch {i}/{num_images}')
                num_items = min(self.parallel_parameters.batch_size, num_images - i)
                img_chunk = img[i:i+num_items]
                img_crop = self.cropper.crop(image=img_chunk)
                e_chunk = self.embedding.extract(image=img_crop)
                if detach:
                    e_chunk = e_chunk.detach()
                e[i:i+num_items] = e_chunk
        if detach:
            e = e.detach()
        return e

    # ---

    def embedding_from_w(
            self,
            w : torch.Tensor,
            c : torch.Tensor | None,
            detach : bool = True
            ) -> torch.Tensor:
        """ Extract embedding from an image (batch). """
        num_latents = w.shape[0]
        if c is not None:
            assert c.ndim == 2
            assert c.shape[0] == num_latents
            assert c.shape[1] == 25
        if num_latents <= self.parallel_parameters.batch_size:
            img = self.generator.synthesis(w=w, c=c)
            img_crop = self.cropper.crop(image=img)
            e = self.embedding.extract(image=img_crop)
        else:
            e = torch.empty(
                (num_latents, self.embedding.e_dim),
                device=self.device,
                dtype=self.dtype)
            for i in tqdm(
                    range(0, num_latents, self.parallel_parameters.batch_size),
                    disable=self.disable_progress_bar):
                num_items = min(self.parallel_parameters.batch_size, num_latents - i)
                w_chunk = w[i:i+num_items]
                c_chunk = c[i:i+num_items] if c is not None else None
                img_chunk = self.generator.synthesis(w=w_chunk, c=c_chunk)
                img_crop = self.cropper.crop(image=img_chunk)
                e_chunk = self.embedding.extract(image=img_crop)
                if detach:
                    e_chunk = e_chunk.detach()
                    del img_chunk, img_crop
                e[i:i+num_items] = e_chunk
        if detach:
            e = e.detach()
        return e

    # ---

    @dataclass
    class EmbeddingStats:
        dist_min : float = float('nan')
        dist_avg : float = float('nan')
        dist_max : float = float('nan')
        prop_ict : float = float('nan')

        def echo(
                self,
                fg : str = 'red',
                bold : bool = True) -> None:
            click.secho((f'dist_min={self.dist_min:.3f} '
                         f'dist_avg={self.dist_avg:.3f} '
                         f'dist_max={self.dist_max:.3f}'), fg=fg, bold=bold)
            click.secho(f'prop_ict={self.prop_ict:.3f}', fg=fg, bold=bold)

        @staticmethod
        def create_h5_datasets(
                stat_file : h5py.File,
                num_iterations : int) -> None:
            dset_shape = (num_iterations, )
            stat_file.create_dataset(name='dist_min', shape=dset_shape)
            stat_file.create_dataset(name='dist_max', shape=dset_shape)
            stat_file.create_dataset(name='dist_avg', shape=dset_shape)
            stat_file.create_dataset(name='prop_ict', shape=dset_shape)

        def write_to_h5(
                self,
                stat_file : h5py.File,
                iter : int) -> None:
            stat_file['dist_min'][iter] = self.dist_min
            stat_file['dist_max'][iter] = self.dist_max
            stat_file['dist_avg'][iter] = self.dist_avg
            stat_file['prop_ict'][iter] = self.prop_ict

    # TODO ndim 3 inter-intra
    def compute_embedding_stats(
            self,
            embeddings : torch.Tensor,
            ict : float = 0.0,
            block_size : int = 500,
            ) -> EmbeddingStats:
        """
            Compute inter-embedding distance statistics.

            embeddings : torch.Tensor of dim 2 or 3, last dimension is embedding dimension
            single: bool if False this algorithm double counts every entry, use for performance
            block_size: int value for the size of the block.
            returns: an EmbeddingStats dataclass
        """
        embedding_stats = self.EmbeddingStats()
        if embeddings.ndim == 2:
            assert embeddings.shape[1] == self.embedding.e_dim
            num_embeddings = embeddings.shape[0]
        elif embeddings.ndim == 3:
            assert embeddings.shape[2] == self.embedding.e_dim
            num_embeddings = embeddings.shape[0] * embeddings.shape[1]
            # TEMP re-arange as 2D
            # TODO separate intra class and interclass var
            embeddings = einops.rearrange(embeddings, 'i a e -> (i a) e')
        else:
            raise RuntimeError('Wrong embedding dimensions')
        if num_embeddings <= block_size:
            if embeddings.ndim == 2:
                distances = self.embedding.distance(embeddings, embeddings)
                n = num_embeddings
                # keep elements above the diagonal
                tri_up_idx = torch.triu_indices(n, n, offset=1, dtype=torch.long, device=self.device)
                tri_up = distances[tri_up_idx[0], tri_up_idx[1]]
                embedding_stats.dist_min = torch.min(tri_up)
                embedding_stats.dist_max = torch.max(tri_up)
                embedding_stats.dist_avg = torch.mean(tri_up)
                bigger_ict = torch.where(tri_up > ict, 1.0, 0.0)
                embedding_stats.prop_ict = torch.sum(bigger_ict) / tri_up.shape[0]
            else: # ndim == 3
                raise RuntimeError('Not yet implemented')
        else:
            if embeddings.ndim == 3:
                raise RuntimeError('Not yet implemented')
            num_blocks = num_embeddings // block_size
            embedding_stats.dist_min = float('inf')
            embedding_stats.dist_max = float('-inf')
            embedding_stats.dist_avg = 0.0
            num_elements = 0
            num_above_ict = 0
            for i in range(num_blocks):
                i0 = i * block_size
                i1 = min((i + 1) * block_size, num_embeddings)
                n_i = i1 - i0
                e_i = embeddings[i0 : i1, :]
                for j in range(i, num_blocks):
                    j0 = j * block_size
                    j1 = min((j + 1) * block_size, num_embeddings)
                    n_j = j1 - j0
                    e_j = embeddings[j0: j1, :]
                    dist_ij = self.embedding.distance(e_i, e_j)
                    if i == j:
                        tri_up_idx = torch.triu_indices(n_i, n_j, offset=1, dtype=torch.long, device=self.device)
                        tri_up = dist_ij[tri_up_idx[0], tri_up_idx[1]]
                        dist_min = torch.min(tri_up)
                        dist_max = torch.max(tri_up)
                        embedding_stats.dist_avg += torch.sum(tri_up)
                        num_elements += tri_up.shape[0]
                        bigger_ict = torch.where(tri_up > ict, 1.0, 0.0)
                        num_above_ict += int(torch.sum(bigger_ict))
                    else: # j > i
                        dist_min = torch.min(dist_ij)
                        dist_max = torch.max(dist_ij)
                        embedding_stats.dist_avg += torch.sum(dist_ij)
                        num_elements += dist_ij.shape[0] * dist_ij.shape[1]
                        bigger_ict = torch.where(dist_ij > ict, 1.0, 0.0)
                        num_above_ict += int(torch.sum(bigger_ict))
                    embedding_stats.dist_min = min(dist_min, embedding_stats.dist_min)
                    embedding_stats.dist_max = max(dist_max, embedding_stats.dist_max)
            embedding_stats.dist_avg = embedding_stats.dist_avg / num_elements
            embedding_stats.prop_ict = float(num_above_ict) / float(num_elements)
        # pull back to CPU
        embedding_stats.dist_min = float(embedding_stats.dist_min)
        embedding_stats.dist_avg = float(embedding_stats.dist_avg)
        embedding_stats.dist_max = float(embedding_stats.dist_max)
        embedding_stats.prop_ict = float(embedding_stats.prop_ict)
        return embedding_stats

    # ---

    @dataclass
    class LatentStats:
        w2wd_min : float = float('nan')
        w2wd_avg : float = float('nan')
        w2wd_max : float = float('nan')
        wavg_avg : float = float('nan')

        def echo(
                self,
                fg : str = 'green',
                bold : bool = True) -> None:
            click.secho((f'w2wd_min={self.w2wd_min:.3f} '
                         f'w2wd_avg={self.w2wd_avg:.3f} '
                         f'w2wd_max={self.w2wd_max:.3f}'), fg=fg, bold=bold)
            click.secho(f'wavg_avg={self.wavg_avg:.3f}', fg=fg, bold=bold)

        @staticmethod
        def create_h5_datasets(
                stat_file : h5py.File,
                num_iterations : int) -> None:
            dset_shape = (num_iterations, )
            stat_file.create_dataset(name='w2wd_min', shape=dset_shape)
            stat_file.create_dataset(name='w2wd_avg', shape=dset_shape)
            stat_file.create_dataset(name='w2wd_max', shape=dset_shape)
            stat_file.create_dataset(name='wavg_avg', shape=dset_shape)

        def write_to_h5(
                self,
                stat_file : h5py.File,
                iter : int) -> None:
            stat_file['w2wd_min'][iter] = self.w2wd_min
            stat_file['w2wd_avg'][iter] = self.w2wd_avg
            stat_file['w2wd_max'][iter] = self.w2wd_max
            stat_file['wavg_avg'][iter] = self.wavg_avg

    def compute_latent_stats(
            self,
            latents : torch.Tensor,
            block_size : int = 500
            ) -> LatentStats:
        """ Compute latent distance statistics. """
        latent_stats = self.LatentStats()
        w_avg = self.generator.w_avg.unsqueeze(0)
        if latents.ndim == 2:
            assert latents.shape[1] == self.generator.w_dim
            num_latents = latents.shape[0]
        elif latents.ndim == 3:
            assert latents.shape[2] == self.generator.w_dim
            num_latents = latents.shape[0] * latents.shape[1]
            # TEMP re-arange as 2D
            # TODO separate intra class and interclass var
            latents = einops.rearrange(latents, 'i a w -> (i a) w')
        else:
            raise RuntimeError('Wrong latents dimensions')
        if num_latents <= block_size:
            if latents.ndim == 2:
                distances = torch.cdist(latents, latents)
                n = num_latents
                # keep elements above the diagonal
                tri_up_idx = torch.triu_indices(n, n, offset=1, dtype=torch.long, device=self.device)
                tri_up = distances[tri_up_idx[0], tri_up_idx[1]]
                latent_stats.w2wd_min = torch.min(tri_up)
                latent_stats.w2wd_max = torch.max(tri_up)
                latent_stats.w2wd_avg = torch.mean(tri_up)
                w_avg_distances = torch.cdist(latents, w_avg)
                latent_stats.wavg_avg = torch.mean(w_avg_distances)
            else: # ndim == 3
                raise RuntimeError('Not yet implemented')
        else:
            if latents.ndim == 3:
                raise RuntimeError('Not yet implemented')
            num_blocks = num_latents // block_size
            latent_stats.w2wd_min = float('inf')
            latent_stats.w2wd_max = float('-inf')
            latent_stats.w2wd_avg = 0.0
            # w avg dist (no need to batch)
            w_avg_distances = torch.cdist(latents, w_avg)
            latent_stats.wavg_avg = torch.mean(w_avg_distances)
            num_elements = 0
            for i in range(num_blocks):
                i0 = i * block_size
                i1 = min((i + 1) * block_size, num_latents)
                n_i = i1 - i0
                w_i = latents[i0 : i1, :]
                for j in range(i, num_blocks):
                    j0 = j * block_size
                    j1 = min((j + 1) * block_size, num_latents)
                    n_j = j1 - j0
                    w_j = latents[j0: j1, :]
                    dist_ij = torch.cdist(w_i, w_j)
                    if i == j:
                        tri_up_idx = torch.triu_indices(n_i, n_j, offset=1, dtype=torch.long, device=self.device)
                        tri_up = dist_ij[tri_up_idx[0], tri_up_idx[1]]
                        w2wd_min = torch.min(tri_up)
                        w2wd_max = torch.max(tri_up)
                        latent_stats.w2wd_avg += torch.sum(tri_up)
                        num_elements += tri_up.shape[0]
                    else: # j > i
                        w2wd_min = torch.min(dist_ij)
                        w2wd_max = torch.max(dist_ij)
                        latent_stats.w2wd_avg += torch.sum(dist_ij)
                        num_elements += dist_ij.shape[0] * dist_ij.shape[1]
                    latent_stats.w2wd_min = min(w2wd_min, latent_stats.w2wd_min)
                    latent_stats.w2wd_max = max(w2wd_max, latent_stats.w2wd_max)
            latent_stats.w2wd_avg = latent_stats.w2wd_avg / num_elements
        # pull back to CPU
        latent_stats.w2wd_min = float(latent_stats.w2wd_min)
        latent_stats.w2wd_avg = float(latent_stats.w2wd_avg)
        latent_stats.w2wd_max = float(latent_stats.w2wd_max)
        latent_stats.wavg_avg = float(latent_stats.wavg_avg)
        return latent_stats

    # ---

    @dataclass
    class ForceStats:
        intf_min : float = float('nan')
        intf_avg : float = float('nan')
        intf_max : float = float('nan')
        rndf_min : float = float('nan')
        rndf_avg : float = float('nan')
        rndf_max : float = float('nan')
        latf_min : float = float('nan')
        latf_avg : float = float('nan')
        latf_max : float = float('nan')
        totf_min : float = float('nan')
        totf_avg : float = float('nan')
        totf_max : float = float('nan')

        def echo(
                self,
                fg : str = 'blue',
                bold : bool = True) -> None:
            click.secho((f'intf_min={self.intf_min:.3f} '
                         f'intf_avg={self.intf_avg:.3f} '
                         f'intf_max={self.intf_max:.3f}'), fg=fg, bold=bold)
            click.secho((f'rndf_min={self.rndf_min:.3f} '
                         f'rndf_avg={self.rndf_avg:.3f} '
                         f'rndf_max={self.rndf_max:.3f}'), fg=fg, bold=bold)
            click.secho((f'latf_min={self.latf_min:.3f} '
                         f'latf_avg={self.latf_avg:.3f} '
                         f'latf_max={self.latf_max:.3f}'), fg=fg, bold=bold)
            click.secho((f'totf_min={self.totf_min:.3f} '
                         f'totf_avg={self.totf_avg:.3f} '
                         f'totf_max={self.totf_max:.3f}'), fg=fg, bold=bold)

        @staticmethod
        def create_h5_datasets(
                stat_file : h5py.File,
                num_iterations : int) -> None:
            dset_shape = (num_iterations, )
            stat_file.create_dataset(name='intf_min', shape=dset_shape)
            stat_file.create_dataset(name='intf_avg', shape=dset_shape)
            stat_file.create_dataset(name='intf_max', shape=dset_shape)
            stat_file.create_dataset(name='rndf_min', shape=dset_shape)
            stat_file.create_dataset(name='rndf_avg', shape=dset_shape)
            stat_file.create_dataset(name='rndf_max', shape=dset_shape)
            stat_file.create_dataset(name='latf_min', shape=dset_shape)
            stat_file.create_dataset(name='latf_avg', shape=dset_shape)
            stat_file.create_dataset(name='latf_max', shape=dset_shape)
            stat_file.create_dataset(name='totf_min', shape=dset_shape)
            stat_file.create_dataset(name='totf_avg', shape=dset_shape)
            stat_file.create_dataset(name='totf_max', shape=dset_shape)

        def write_to_h5(
                self,
                stat_file : h5py.File,
                iter : int) -> None:
            stat_file['intf_min'][iter] = self.intf_min
            stat_file['intf_avg'][iter] = self.intf_avg
            stat_file['intf_max'][iter] = self.intf_max
            stat_file['rndf_min'][iter] = self.rndf_min
            stat_file['rndf_avg'][iter] = self.rndf_avg
            stat_file['rndf_max'][iter] = self.rndf_max
            stat_file['latf_min'][iter] = self.latf_min
            stat_file['latf_avg'][iter] = self.latf_avg
            stat_file['latf_max'][iter] = self.latf_max
            stat_file['totf_min'][iter] = self.totf_min
            stat_file['totf_avg'][iter] = self.totf_avg
            stat_file['totf_max'][iter] = self.totf_max

    def compute_force_stats(
            self,
            interaction_force : torch.Tensor,
            random_force : torch.Tensor,
            latent_force : torch.Tensor,
            total_force : torch.Tensor
            ) -> ForceStats:
        """ Compute force statistics. """
        force_stats = self.ForceStats()
        interaction_force_norm = torch.linalg.vector_norm(interaction_force, dim=1)
        force_stats.intf_min = float(torch.min(interaction_force_norm))
        force_stats.intf_avg = float(torch.mean(interaction_force_norm))
        force_stats.intf_max = float(torch.max(interaction_force_norm))
        random_force_norm = torch.linalg.vector_norm(random_force, dim=1)
        force_stats.rndf_min = float(torch.min(random_force_norm))
        force_stats.rndf_avg = float(torch.mean(random_force_norm))
        force_stats.rndf_max = float(torch.max(random_force_norm))
        latent_force_norm = torch.linalg.vector_norm(latent_force, dim=1)
        force_stats.latf_min = float(torch.min(latent_force_norm))
        force_stats.latf_avg = float(torch.mean(latent_force_norm))
        force_stats.latf_max = float(torch.max(latent_force_norm))
        total_force_norm = torch.linalg.vector_norm(total_force, dim=1)
        force_stats.totf_min = float(torch.min(total_force_norm))
        force_stats.totf_avg = float(torch.mean(total_force_norm))
        force_stats.totf_max = float(torch.max(total_force_norm))
        return force_stats

    # ---

    @dataclass
    class TimingStats:
        time_total : float = float('nan')
        time_iter : float = float('nan')
        num_candidates : int = 0

        def echo(
                self,
                fg : str = 'magenta',
                bold : bool = True) -> None:
            click.secho((f'time_tot={self.time_total:.3f} '
                         f'time_itr={self.time_iter:.3f}'), fg=fg, bold=bold)

        @staticmethod
        def create_h5_datasets(
                stat_file : h5py.File,
                num_iterations : int) -> None:
            dset_shape = (num_iterations, )
            stat_file.create_dataset(name='time_total', shape=dset_shape)
            stat_file.create_dataset(name='time_iter', shape=dset_shape)

        def write_to_h5(
                self,
                stat_file : h5py.File,
                iter : int) -> None:
            stat_file['time_total'][iter] = self.time_total
            stat_file['time_iter'][iter] = self.time_iter

    def compute_timing_stats(
            self,
            start_time : float,
            last_time : float,
            current_time : float,
            ) -> TimingStats:
        """
            Compute timing statistics.
        """
        timing_stats = self.TimingStats()
        timing_stats.time_total = current_time - start_time
        timing_stats.time_iter = current_time - last_time
        return timing_stats

    # ---

    def merge_stats_files(
            self,
            remove : bool = False
            ) -> None:
        """ Merge stats files in the root directory. """
        click.echo('Merging stats files...')
        stats_files_paths = glob.glob(os.path.join(self.root_directory, 'stats_*.h5'))
        click.echo(f'Found {len(stats_files_paths)} files, loading ...')
        stats_files : list[h5py.File] = []
        dataset_names : list[str] | None = None
        def close_files():
            for stats_file in stats_files:
                try:
                    stats_file.close()
                except:
                    pass
        try:
            for stats_files_path in stats_files_paths:
                stats_file = h5py.File(stats_files_path, mode='r')
                stats_files.append(stats_file)
                if dataset_names is None:
                    dataset_names = sorted(stats_file.keys())
                elif sorted(stats_file.keys()) == dataset_names:
                    pass # assets keys match in stats files
                else:
                    click.secho('Key mismatch, exiting...', fg='red')
                    raise Exception
        except:
            close_files()
            return
        merged_file_path = os.path.join(self.root_directory, 'stats.h5')
        if os.path.exists(merged_file_path):
            click.secho('File already exists, exiting...')
            close_files()
            return
        with h5py.File(merged_file_path, 'w') as merged_file:
            try:
                device = torch.device('cpu')
                for dataset_name in dataset_names:
                    dataset_len : int | None = None
                    dataset_dtype : torch.dtype | None = None
                    numpy_dtype : np.dtype = None
                    datasets = []
                    for stats_file in stats_files:
                        dataset = torch.tensor(stats_file[dataset_name], device=device)
                        if dataset_len is None:
                            dataset_len = dataset.shape[0]
                        elif dataset.shape[0] != dataset_len:
                            raise RuntimeError('Dataset length missmatch, exiting...')
                        if dataset_dtype is None:
                            dataset_dtype = dataset.dtype
                            numpy_dtype = dataset.numpy().dtype
                        elif dataset.dtype != dataset_dtype:
                            raise RuntimeError('Dataset type missmatch, exiting...')
                        datasets.append(dataset)
                    dataset = merged_file.create_dataset(
                        name=dataset_name,
                        shape=(dataset_len,),
                        dtype=numpy_dtype)
                    datasets = torch.stack(datasets)
                    if dataset_name.endswith('_min'):
                        dataset[:] = datasets.amin(dim=0)
                    elif dataset_name.endswith('_avg'):
                        dataset[:] = datasets.mean(dim=0)
                    elif dataset_name.endswith('_max'):
                        dataset[:] = datasets.amax(dim=0)
                    else:
                        click.secho('Cannot infer dataset postfix...')
            except:
                close_files()
                return
            close_files()
            if remove:
                for path in stats_files_paths:
                    os.remove(path=path)

    # ---

    def memory_check(self):
        alloc = float(torch.cuda.memory_allocated())/1000000000
        max_alloc = float(torch.cuda.max_memory_allocated())/1000000000
        click.secho(f'alloc = {alloc:.3f} GB, max_alloc = {max_alloc:.3f} GB')

    # ---

    def create_references_random(
            self,
            references_parameters : ReferencesParameters
            ) -> None:
        """
            Create new identities by randomly sampling the latent space.
            This algorithm performs no checks for the references to be
            distant enough.
        """
        assert not self.parallel_parameters.job_parallel
        num_identities = len(self.identities)
        self.torch_seed(self.identities[0])
        click.echo('Generating Z latents...')
        z = self.generate_z_latents(num_latents=num_identities)
        click.echo('Generating W latents...')
        if self.network_type == 'eg3d':
            camera = self.generator.camera(fov_deg=references_parameters.camera_fov)
            c = einops.repeat(camera, '1 c -> n c', n=num_identities)
        else:
            c = None
        w = self.map_z_to_w(z=z, c=c)
        if references_parameters.neutralisation:
            click.echo('Neutralizing W latents...')
            w = self.latent_edit.neutralisation(w)
        click.echo('Generating images...')
        img = self.generate_image_from_w(w=w, c=c)
        click.echo('Generating embeddings...')
        e = self.extract_embedding_from_image(img=img)
        click.echo('Saving samples...')
        self.save_to_collection(
            collection=self.sample_collection,
            identities=self.identities,
            latents=w,
            camera=c,
            z_latents=z,
            embeddings=e)
        if self.generate_images:
            click.echo('Saving images...')
            self.save_images(
                identities=self.identities,
                latents=w,
                cameras=c)
        click.echo('...Done')

    # ---

    def create_references_reject_repulse(
            self,
            compared_embeddings : torch.Tensor,
            repulse : bool,
            references_parameters : ReferencesParameters
            ) -> None:
        """
            Create new identities by iteratively sampling the latent space,
            the new identity's embedding is compared to a list of previously
            created ones and accepted only if it is distant enough. In the
            repulse mode the w vector is updated to 'repulse' it from
            other identities.
        """
        assert not self.parallel_parameters.job_parallel
        num_new_identities = len(self.identities)
        num_previous_identities = compared_embeddings.shape[0]
        start_time = None
        t_stats = self.TimingStats()
        stat_file_path = os.path.join(self.root_directory, 'stats.h5')
        if os.path.exists(stat_file_path):
            os.remove(stat_file_path)
        with h5py.File(stat_file_path, 'w') as stat_file:
            self.TimingStats.create_h5_datasets(
                stat_file=stat_file,
                num_iterations=num_new_identities)
            stat_file.flush()
        embeddings = torch.empty(
            (num_previous_identities + num_new_identities, self.embedding.e_dim),
            device=self.device,
            dtype=self.dtype)
        if self.network_type == 'eg3d':
            camera = self.generator.camera(fov_deg=references_parameters.camera_fov)
        else:
            camera = None
        if num_previous_identities > 0:
            embeddings[num_previous_identities, :] = compared_embeddings
        avg_w = self.generator.w_avg.unsqueeze(0)
        for idx, identity in enumerate(self.identities):
            click.echo('---------------------------------------------------------------------')
            click.echo('Identity {} ({}/{})'.format(identity, idx + 1, num_new_identities))
            current_time = time.time()
            if start_time is None:
                start_time = current_time
            num_existing = num_previous_identities + idx
            existing_embeddings = embeddings[0:num_existing, :]
            accepted = False
            num_candidates = 0
            while not accepted:
                num_candidates += 1
                self.torch_seed(idx + num_candidates * num_new_identities)
                z = self.generate_z_latents(num_latents=1)
                w = self.map_z_to_w(z=z, c=camera)
                if not self.disable_progress_bar:
                    w_avg_dist_0 = torch.mean(torch.cdist(w, avg_w))
                    click.echo(f'w_avg_dist_0: {w_avg_dist_0:.3f}')
                for _ in range(references_parameters.max_iterations_repulse):
                    if references_parameters.neutralisation:
                        w = self.latent_edit.neutralisation(w)
                    if repulse:
                        w = w.clone().detach().requires_grad_(True)
                    img = self.generate_image_from_w(w=w, c=camera, detach=False)
                    e = self.extract_embedding_from_image(img=img, detach=False)
                    if existing_embeddings.shape[0] == 0:
                        accepted = True
                        break
                    distances = self.embedding.distance(e, existing_embeddings)
                    distances = distances[0] # discard dummy index
                    w_avg_dist = torch.mean(torch.cdist(w, avg_w))
                    if not self.disable_progress_bar:
                        click.echo(f'dist: max: {torch.max(distances):.3f} min: {torch.min(distances):.3f} w_avg_dist: {w_avg_dist:.3f}')
                    ict = references_parameters.ict
                    if torch.all(distances >= ict):
                        accepted = True
                        break
                    if not repulse:
                        break
                    repulse = references_parameters.repulse
                    interactions = torch.where(distances < ict, repulse * (ict - distances)**2/2, 0.0)
                    potential = torch.sum(interactions)
                    potential.backward()
                    force = -w.grad
                    with torch.no_grad():
                        w = w + force
                    if references_parameters.rescale_latents:
                       w = avg_w +  (avg_w - w) * (w_avg_dist_0 / w_avg_dist)
            e = e.detach()
            embeddings[idx, :] = e[0]
            time_now = time.time()
            t_stats.time_iter = time_now - current_time
            current_time = time_now
            t_stats.time_total = time_now - start_time
            t_stats.num_candidates = num_candidates
            with h5py.File(stat_file_path, 'a') as stat_file:
                t_stats.write_to_h5(stat_file=stat_file, iter=idx)
            if repulse:
                z = None
            sample = Sample(
                z_latent=z,
                w_latent=w,
                c_label=camera,
                embedding=e,
                network_type=self.network_type,
                embedding_type=self.embedding_type)
            self.sample_collection.add_sample(
                identity=identity,
                label='reference',
                sample=sample)
            if self.generate_images:
                image_post = self.postprocess_image(img=img)
                utils.save_image(
                    image=image_post,
                    file_path=self.get_image_path(identity, 'reference'),
                    create_directories=True)
        self.sample_collection.save()

    # ---

    def create_references_langevin(
            self,
            references_parameters : ReferencesParameters,
            langevin_parameters : LangevinParameters
            ) -> None:
        """ Create a set of new identities using the Langevin algorithm. """
        assert not self.parallel_parameters.job_parallel
        num_identities = len(self.identities)
        self.torch_seed(self.identities[0])
        if self.network_type == 'eg3d':
            camera = self.generator.camera(fov_deg=references_parameters.camera_fov)
            c = einops.repeat(camera, '1 c -> n c', n=num_identities)
        else:
            c = None
        if langevin_parameters.constant_embeddings_path is not None:
            click.echo('Loading constant embeddings...')
            constant_embeddings_collection = SampleCollection(
                file_path=langevin_parameters.constant_embeddings_path,
                read_only=True)
            constant_embeddings_collection.load(
                device=self.device,
                dtype=self.dtype)
            constant_identities = constant_embeddings_collection.list_identities()
            constant_id = []
            constant_label = []
            for identity in constant_identities:
                labels = constant_embeddings_collection.list_identity_labels(identity=identity)
                if labels is not None and len(labels) > 0:
                    constant_id.append(identity)
                    constant_label.append(labels[0])
            num_constant_identities = len(constant_id)
            click.echo(f'Loading {num_constant_identities} embeddings...')
            constant_embeddings = torch.empty(
                (num_constant_identities, self.embedding.e_dim),
                device=self.device,
                dtype=self.dtype)
            for i in range(num_constant_identities):
                sample = constant_embeddings_collection.get_sample(
                    identity=constant_id[i],
                    label=constant_label[i])
                assert sample.embedding_type == self.embedding_type
                constant_embeddings[i, :] = sample.embedding[0, :]
            constant_embeddings = constant_embeddings.detach().requires_grad_(False)
        else:
            constant_embeddings = None
        root_directory_files = os.listdir(self.root_directory)
        if 'samples.h5' in root_directory_files:
            click.echo('Sample collection already present in root directory, exiting...')
            return
        langevin_checkpoints = []
        for file in root_directory_files:
            if file.endswith('.h5') and file.startswith('langevin_checkpoint_'):
                langevin_checkpoints.append(file)
        if len(langevin_checkpoints) > 0:
            resume_run = True
            checkpoint_file = langevin_checkpoints[-1]
            click.echo(f'Loading checkpoint: {checkpoint_file}')
            start_iteration = int(checkpoint_file.split('_')[2].split('.')[0])
            click.echo(f'Resuming at iteration: {start_iteration}')
            checkpoint_path = os.path.join(self.root_directory, checkpoint_file)
            try:
                checkpoint = SampleCollection(file_path=checkpoint_path, read_only=True)
                checkpoint.load(device=self.device, dtype=self.dtype)
                assert checkpoint.list_identities() == self.identities, \
                    'The checkpoint does not contain the correct number of identity'
            except Exception as e:
                raise RuntimeError(f'Error while loading checkpoint {checkpoint_path} : {str(e)}')
        else:
            start_iteration = 0
            checkpoint = None
            resume_run = False
        stat_file_path = os.path.join(self.root_directory, 'stats.h5')
        if not resume_run:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)
            with h5py.File(stat_file_path, 'w') as stat_file:
                dset_shape = (langevin_parameters.iterations, )
                self.EmbeddingStats.create_h5_datasets(stat_file, langevin_parameters.iterations)
                self.LatentStats.create_h5_datasets(stat_file, langevin_parameters.iterations)
                self.ForceStats.create_h5_datasets(stat_file, langevin_parameters.iterations)
                stat_file.create_dataset(name='timestep', shape=dset_shape)
                stat_file.flush()
            click.echo('Generating initial latents...')
            z_all = self.generate_z_latents(num_latents=num_identities)
            w_all = self.map_z_to_w(z=z_all, c=c)
        else: # resume run
            click.echo('Importing latents from checkpoint...')
            w_all = torch.empty(
                (num_identities, self.generator.w_dim),
                device=self.device,
                dtype=self.dtype)
            for idx, identity in enumerate(self.identities):
                sample = checkpoint.get_sample(identity=identity, label='reference')
                assert sample.network_type == self.network_type
                w_all[idx, :] = sample.w_latent[0, :]
        interaction_force = torch.empty(
            (num_identities, self.generator.w_dim),
            device=self.device,
            dtype=self.dtype)
        self.memory_check()
        for iter in range(start_iteration, langevin_parameters.iterations + 1):
            click.echo(f'iter = {iter}')
            w_all = w_all.clone().detach().requires_grad_(False)
            click.echo('Generating fixed embeddings...')
            embeddings = self.embedding_from_w(w=w_all, c=c, detach=True)
            self.memory_check()
            click.echo('Generating stats...')
            e_stats = self.compute_embedding_stats(
                embeddings=embeddings,
                ict=references_parameters.ict)
            e_stats.echo()
            if iter == langevin_parameters.iterations:
                break
            click.echo('Computing interactions...')
            for i in tqdm(range(num_identities), disable=self.disable_progress_bar):
                w_i = w_all[i].unsqueeze(0).clone().detach().requires_grad_(True)
                c_i = c[i].unsqueeze(0) if c is not None else None
                img_i = self.generate_image_from_w(w=w_i, c=c_i, detach=False)
                e_i = self.extract_embedding_from_image(
                    img=img_i,
                    detach=False)
                e_others = torch.cat((embeddings[0:i, :], embeddings[i+1:, :]))
                dist_others = self.embedding.distance(e_i, e_others)
                if torch.any(torch.isnan(dist_others)):
                    raise RuntimeError(f'Found NaN: {i}')
                dist_others = dist_others[0]
                k = langevin_parameters.repulsion_coefficient
                d_0 = langevin_parameters.repulsion_radius
                interactions = torch.where(dist_others < d_0, k * (d_0 - dist_others)**2 / 2, 0.0)
                potential = torch.sum(interactions)
                if constant_embeddings is not None:
                    dist_cst_e = self.embedding.distance(e_i, constant_embeddings)
                    d_0_cst = langevin_parameters.constant_distance
                    interactions_cst = torch.where(dist_cst_e < d_0_cst, k * (d_0_cst - dist_cst_e)**2 / 2, 0.0)
                    potential += torch.sum(interactions_cst)
                potential.backward()
                interaction_force_i = -w_i.grad
                interaction_force[i] = interaction_force_i.detach()[0]
            click.echo('Updating latents...')
            with torch.no_grad():
                k_l = langevin_parameters.latent_force
                w_avg = einops.repeat(self.generator.w_avg, 'w -> n w', n=w_all.shape[0])
                latent_force = k_l * (w_avg - w_all)
                total_force = interaction_force + latent_force
                eta_0 = langevin_parameters.random_force
                random_force = eta_0 * torch.randn(w_all.shape, device=self.device, dtype=self.dtype)
                f_stats = self.compute_force_stats(
                    interaction_force=interaction_force,
                    random_force=random_force,
                    latent_force=latent_force,
                    total_force=total_force)
                f_stats.echo()
                w_stats = self.compute_latent_stats(latents=w_all)
                w_stats.echo()
                if langevin_parameters.timestep < 0.0:
                    tau_dt = - langevin_parameters.timestep
                    dt = tau_dt * langevin_parameters.viscous_coefficient \
                        * w_stats.w2wd_min / f_stats.totf_max
                    click.secho(f'dt={dt}', fg='green', bold=True)
                else:
                    dt = langevin_parameters.timestep
                mu = langevin_parameters.viscous_coefficient
                dw = (dt / mu) * total_force + (math.sqrt(dt) / mu) * random_force
                w_all = w_all + dw
                with h5py.File(stat_file_path, 'a') as stat_file:
                    e_stats.write_to_h5(stat_file=stat_file, iter=iter)
                    w_stats.write_to_h5(stat_file=stat_file, iter=iter)
                    f_stats.write_to_h5(stat_file=stat_file, iter=iter)
                    stat_file['timestep'][iter] = float(dt)
                if langevin_parameters.checkpoint_iter > 0 and \
                        iter % langevin_parameters.checkpoint_iter == 0:
                    click.echo('Saving checkpoint...')
                    collection_name = f'langevin_checkpoint_{iter:04}.h5'
                    collection_path = os.path.join(self.root_directory, collection_name)
                    collection = SampleCollection(collection_path, read_only=False)
                    self.save_to_collection(
                        collection=collection,
                        identities=self.identities,
                        latents=w_all,
                        embeddings=embeddings,
                        camera=c)
        click.echo('Saving samples...')
        self.save_to_collection(
            collection=self.sample_collection,
            identities=self.identities,
            latents=w_all,
            embeddings=embeddings,
            camera=c)
        if self.generate_images:
            click.echo('Saving images...')
            self.save_images(
                identities=self.identities,
                latents=w_all,
                cameras=c)
        click.echo('...Done')

    # ---

    def create_references(
            self,
            algorithm : str,
            references_parameters : ReferencesParameters,
            langevin_parameters : LangevinParameters
            ) -> None:
        """ Create all references for the provided identity tags. """
        click.echo('Creating references...')
        assert not self.parallel_parameters.job_parallel
        previous_identities = self.sample_collection.list_identities()
        if previous_identities is not None:
            if len(previous_identities) >= len(self.identities):
                click.echo('All done! Exiting...')
                return
            elif len(previous_identities) > 0:
                click.echo(f'Loading {len(previous_identities)} existing identities...')
            else:
                pass
        compared_embeddings = torch.empty(
            (len(previous_identities), self.embedding.e_dim),
            device=self.device,
            dtype=self.dtype)
        for i, identity in zip(range(len(previous_identities)), previous_identities):
            labels = self.sample_collection.list_identity_labels(identity=identity)
            if 'reference' in labels:
                reference_sample = self.sample_collection.get_sample(identity, 'reference')
                e = reference_sample.embedding
                assert e.shape == (1, self.embedding.e_dim)
                compared_embeddings[i, :] = e[0, :]
            else:
                raise RuntimeError(f'Cannot find reference {identity} in sample file.')
        # Don't regenerate preexisting identities
        self.identities = list(set(self.identities).difference(set(previous_identities)))
        click.echo(f'Will generate {len(self.identities)} new identities...')
        if algorithm == 'random':
            self.create_references_random(
                references_parameters=references_parameters)
        elif algorithm == 'reject':
            self.create_references_reject_repulse(
                references_parameters=references_parameters,
                compared_embeddings=compared_embeddings,
                repulse=False)
        elif algorithm == 'repulse':
            self.create_references_reject_repulse(
                references_parameters=references_parameters,
                compared_embeddings=compared_embeddings,
                repulse=True)
        elif algorithm == 'langevin':
            self.create_references_langevin(
                references_parameters=references_parameters,
                langevin_parameters=langevin_parameters)
        else:
            raise RuntimeError(f'Unknown algorithm: {algorithm}')

    # ---

    def augment_identities_dispersion(
            self,
            dispersion_parameters : DispersionParameters,
            covariates_scaling : LatentEdit.CovariantScaling
            ) -> None:
        """ Augment identities with W repulsion and E attraction. """
        click.echo('Augmenting identities with Dispersion algorithm...')
        if self.parallel_parameters.job_parallel:
            stat_file_name = f'stats_{self.parallel_parameters.job_rank:02}.h5'
        else:
            stat_file_name = 'stats.h5'
        stat_file_path = os.path.join(self.root_directory, stat_file_name)
        # TODO checkpoints and resume
        resume_run = False
        if not resume_run:
            if os.path.exists(stat_file_path):
                os.remove(stat_file_path)
            with h5py.File(stat_file_path, 'w') as stat_file:
                self.EmbeddingStats.create_h5_datasets(
                    stat_file=stat_file,
                    num_iterations=dispersion_parameters.iterations)
                self.LatentStats.create_h5_datasets(
                    stat_file=stat_file,
                    num_iterations=dispersion_parameters.iterations)
                self.ForceStats.create_h5_datasets(
                    stat_file=stat_file,
                    num_iterations=dispersion_parameters.iterations)
                dset_shape = (dispersion_parameters.iterations, )
                stat_file.create_dataset(name='timestep', shape=dset_shape)
                stat_file.flush()
        num_identities = len(self.identities)
        num_augmentations = dispersion_parameters.num_augmentations
        w_dim = self.generator.w_dim
        e_dim = self.embedding.e_dim
        w_ref = torch.empty(
            (num_identities, w_dim),
            device=self.device,
            dtype=self.dtype)
        e_ref = torch.empty(
            (num_identities, e_dim),
            device=self.device,
            dtype=self.dtype)
        w_new = torch.empty(
            (num_identities, num_augmentations, w_dim),
            device=self.device,
            dtype=self.dtype)
        e_new = torch.empty(
            (num_identities, num_augmentations, w_dim),
            device=self.device,
            dtype=self.dtype)
        if self.network_type == 'eg3d':
            c_ref = torch.empty(
                (num_identities, 25),
                device=self.device,
                dtype=self.dtype)
            c_new = torch.empty(
                (num_identities, num_augmentations, 25),
                device=self.device,
                dtype=self.dtype)
        else:
            c_ref = None
            c_new = None
        embedding_force = torch.empty(
            (num_identities, num_augmentations, w_dim),
            device=self.device,
            dtype=self.dtype)
        latent_force = torch.empty(
            (num_identities, num_augmentations, w_dim),
            device=self.device,
            dtype=self.dtype)
        for idx, identity in enumerate(self.identities):
            reference_sample = self.sample_collection.get_sample(identity, 'reference')
            w = reference_sample.w_latent
            e = reference_sample.embedding
            c = reference_sample.c_label
            w_ref[idx, :] = w[0, :]
            e_ref[idx, :] = e[0, :]
            if c_ref is not None:
                c_ref[idx, :] = c[0, :]
        w_new = einops.repeat(w_ref, 'i w -> i a w', a=num_augmentations)
        w_new = w_new.clone()
        if dispersion_parameters.initial_noise > 0.0:
            noise = torch.randn(w_new.shape, device=self.device, dtype=self.dtype) * \
                dispersion_parameters.initial_noise
            w_new += noise
        if dispersion_parameters.initial_covariates:
            if self.latent_edit is None:
                self.latent_edit = LatentEdit(
                    network_type=self.network_type,
                    device=self.device,
                    dtype=self.dtype)
                self.latent_edit.load_covariates_analysis()
                direction_pose = self.latent_edit.pose_normal * \
                    covariates_scaling.pose * \
                    max(-self.latent_edit.pose_neg_mean, self.latent_edit.pose_pos_mean)
                direction_illum = self.latent_edit.illumination_normal * \
                    covariates_scaling.illumination * \
                    max(-self.latent_edit.illumination_neg_mean, self.latent_edit.illumination_pos_mean)
                direction_smile = self.latent_edit.expression_normal['smile'] * \
                    covariates_scaling.expression * \
                    max(-self.latent_edit.expression_neg_mean['smile'],
                         self.latent_edit.expression_pos_mean['smile'])
                direction_scream = self.latent_edit.expression_normal['scream'] * \
                    covariates_scaling.expression * \
                    max(-self.latent_edit.expression_neg_mean['scream'],
                         self.latent_edit.expression_pos_mean['scream'])
                direction_disgust = self.latent_edit.expression_normal['disgust'] * \
                    covariates_scaling.expression * \
                    max(-self.latent_edit.expression_neg_mean['disgust'],
                         self.latent_edit.expression_pos_mean['disgust'])
                direction_squint = self.latent_edit.expression_normal['squint'] * \
                    covariates_scaling.expression * \
                    max(-self.latent_edit.expression_neg_mean['squint'],
                         self.latent_edit.expression_pos_mean['squint'])
                direction_surprise = self.latent_edit.expression_normal['surprise'] * \
                    covariates_scaling.expression * \
                    max(-self.latent_edit.expression_neg_mean['surprise'],
                         self.latent_edit.expression_pos_mean['surprise'])
                self.covariates_directions = torch.stack([ # d x w (d=7)
                    direction_pose,
                    direction_illum,
                    direction_smile,
                    direction_scream,
                    direction_disgust,
                    direction_squint,
                    direction_surprise])
                self.covariates_directions = self.covariates_directions[:, 0, :]
                assert self.covariates_directions.shape[0] == 7
                assert self.covariates_directions.shape[1] == self.generator.w_dim
            if self.covariates_directions is not None:
                num_directions = self.covariates_directions.shape[0]
                direction_weigths = torch.rand( # i x a x d
                    size=(num_identities, num_augmentations, num_directions),
                    device=self.device)
                direction_weigths -= 0.5
                w_new += torch.einsum( # i x a x w
                    'iad, dw-> iaw',
                    direction_weigths,
                    self.covariates_directions)
            else:
                raise RuntimeError('covariates_directions not initialized properly')
        if c_new is not None:
            r = torch.rand((num_identities, num_augmentations), device=self.device, dtype=self.dtype)
            r = torch.sqrt(r)
            a = torch.rand((num_identities, num_augmentations), device=self.device, dtype=self.dtype)
            a = 2.0 * math.pi * a
            pitch = r * torch.cos(a) * dispersion_parameters.camera_angle_variation
            yaw = r * torch.sin(a) * dispersion_parameters.camera_angle_variation
            for i in range(num_identities):
                for a in range(num_augmentations):
                    c_new[i, a, :] = self.generator.camera(
                        fov_deg=dispersion_parameters.camera_fov,
                        pitch_deg=pitch[i, a],
                        yaw_deg=yaw[i, a])[0, :]
        for iter in range(dispersion_parameters.iterations):
            click.echo(f'iter = {iter}')
            w_new = w_new.clone().detach().requires_grad_(False)
            self.memory_check()
            e_stats_global = self.EmbeddingStats(
                dist_min=float('+inf'),
                dist_avg=0.0,
                dist_max=float('-inf'),
                prop_ict=0.0)
            l_stats_global = self.LatentStats(
                w2wd_min=float('+inf'),
                w2wd_avg=0.0,
                w2wd_max=float('-inf'),
                wavg_avg=0.0)
            click.echo('Computing interactions...')
            for i in tqdm(range(num_identities), disable=self.disable_progress_bar):
                e_ref_i = e_ref[i].clone().detach().requires_grad_(False)
                k_e = dispersion_parameters.embedding_coefficient
                if num_augmentations <= self.parallel_parameters.batch_size:
                    w_i = w_new[i].clone().detach().requires_grad_(True)
                    e_i = self.embedding_from_w(
                        w=w_i,
                        c=c_new[i, :, :] if c_new is not None else None,
                        detach=False)
                    dist_e_i = self.embedding.distance(e_i, e_ref_i)
                    embedding_pull_back = k_e * dist_e_i ** 2 / 2.0
                    potential = 0.5 * torch.sum(embedding_pull_back)
                    potential.backward()
                    with torch.no_grad():
                        embedding_force_i = -w_i.grad
                        embedding_force[i] = embedding_force_i.detach()
                else:
                    e_i = torch.empty(
                        size=(num_augmentations, self.embedding.e_dim),
                        device=self.device,
                        dtype=self.dtype,
                        requires_grad=False)
                    for a in range(0, num_augmentations, self.parallel_parameters.batch_size):
                        a_min = a
                        a_max = min(a + self.parallel_parameters.batch_size, num_augmentations)
                        w_ia = w_new[i, a_min:a_max].clone().detach().requires_grad_(True)
                        e_ia = self.embedding_from_w(
                            w=w_ia,
                            c=c_new[i, a_min:a_max, :] if c_new is not None else None,
                            detach=False)
                        dist_e_ia = self.embedding.distance(e_ia, e_ref_i)
                        embedding_pull_back = k_e * dist_e_ia ** 2 / 2.0
                        potential = 0.5 * torch.sum(embedding_pull_back)
                        potential.backward()
                        with torch.no_grad():
                            embedding_force_ia = -w_ia.grad
                            embedding_force_ia = embedding_force_ia[0]
                            embedding_force[i, a_min:a_max] = embedding_force_ia.detach()
                            e_i[a_min:a_max, :] = e_ia[:, :].detach()
                k_w = dispersion_parameters.latent_coefficient
                dw_0 = dispersion_parameters.latent_radius
                w_i = w_new[i].clone().detach().requires_grad_(True)
                dist_w_i = torch.cdist(w_i, w_i)
                overlap_i = dw_0 - dist_w_i
                n = num_augmentations
                off_diag = overlap_i.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1).flatten()
                interactions = torch.where(off_diag > 0, k_w * (off_diag) ** 2 / 4.0, 0.0)
                potential_w = torch.sum(interactions)
                potential_w.backward()
                with torch.no_grad():
                    latent_force_i = -w_i.grad
                with torch.no_grad():
                    k_l = dispersion_parameters.latent_global_coefficient
                    w_avg = einops.repeat(self.generator.w_avg, 'w -> n w', n=w_i.shape[0])
                    latent_force_global = k_l * (w_avg - w_i)
                with torch.no_grad():
                    latent_force[i] = latent_force_i.detach() + latent_force_global
                with torch.no_grad():
                    e_stats = self.compute_embedding_stats(embeddings=e_i)
                    e_stats_global.dist_min = min(e_stats_global.dist_min, float(e_stats.dist_min))
                    e_stats_global.dist_max = max(e_stats_global.dist_max, float(e_stats.dist_max))
                    e_stats_global.dist_avg += float(e_stats.dist_avg)
                    l_stats = self.compute_latent_stats(latents=w_i)
                    l_stats_global.w2wd_min = min(l_stats_global.w2wd_min, float(l_stats.w2wd_min))
                    l_stats_global.w2wd_max = max(l_stats_global.w2wd_max, float(l_stats.w2wd_max))
                    l_stats_global.w2wd_avg += float(l_stats.w2wd_avg)
                    l_stats_global.wavg_avg += float(l_stats.wavg_avg)
            click.echo('Updating latents...')
            with torch.no_grad():
                eta_0 = dispersion_parameters.random_force
                random_force = eta_0 * torch.randn(
                    w_new.shape, device=self.device, dtype=self.dtype)
                total_force = embedding_force + latent_force + random_force
                e_stats_global.dist_avg /= float(num_identities)
                l_stats_global.w2wd_avg /= float(num_identities)
                l_stats_global.wavg_avg /= float(num_identities)
                f_stats_global = self.compute_force_stats(
                    interaction_force=embedding_force,
                    latent_force=latent_force,
                    random_force=random_force,
                    total_force=total_force)
                e_stats_global.echo()
                l_stats_global.echo()
                f_stats_global.echo()
                with h5py.File(stat_file_path, 'a') as stat_file:
                    e_stats_global.write_to_h5(stat_file=stat_file, iter=iter)
                    l_stats_global.write_to_h5(stat_file=stat_file, iter=iter)
                    f_stats_global.write_to_h5(stat_file=stat_file, iter=iter)
                if dispersion_parameters.timestep < 0.0:
                    lambda_dt = - dispersion_parameters.timestep
                    dt = lambda_dt * dispersion_parameters.viscous_coefficient * \
                        l_stats_global.w2wd_min / f_stats_global.totf_max
                    click.secho(f'dt={dt}', fg='green', bold=True)
                else:
                    dt = dispersion_parameters.timestep
                dw = (dt / dispersion_parameters.viscous_coefficient) * total_force
                w_new = w_new + dw
        click.echo('Saving samples...')
        if self.parallel_parameters.job_parallel:
            output_collection_name = f'samples_{self.parallel_parameters.job_rank}.h5'
            output_collection_path = os.path.join(self.root_directory, output_collection_name)
            output_collection = SampleCollection(
                file_path=output_collection_path,
                read_only=False)
        else:
            output_collection = self.sample_collection
        labels = [f'augmentation_{a}' for a in range(num_augmentations)]
        for i in range(num_identities):
            e_new[i, :, :] = self.embedding_from_w(
                w=w_new[i, :, :],
                c=c_new[i, :, :] if c_new is not None else None,
                detach=True)
        self.save_to_collection(
            collection=output_collection,
            identities=self.identities,
            latents=w_new,
            embeddings=e_new,
            camera=c_new,
            label=labels)
        if self.generate_images:
            click.echo('Saving images...')
            self.save_images(
                identities=self.identities,
                latents=w_new,
                cameras=c_new,
                label=labels)
        click.echo('...Done')

    # ---

    def augment_identities_directions(
            self,
            covariates_scaling : LatentEdit.CovariantScaling,
            neutralisation : bool = True
            ) -> None:
        """ Create augmentations from latent directions """
        assert not self.parallel_parameters.job_parallel
        click.echo('Augmenting identities...')
        num_identities = len(self.identities)
        if self.latent_edit is None:
            self.latent_edit = LatentEdit(
                network_type=self.network_type,
                device=self.device,
                dtype=self.dtype)
            self.latent_edit.load_covariates_analysis()
        for i, identity in enumerate(tqdm(self.identities, disable=self.disable_progress_bar)):
            with torch.no_grad():
                click.echo(f'Identity {identity} ({i + 1}/{num_identities})')
                reference_sample = self.sample_collection.get_sample(identity, 'reference')
                w = reference_sample.w_latent # 1 x w
                if neutralisation:
                    w = self.latent_edit.neutralisation(w)
                w_augmented, labels = self.latent_edit.augmentation(
                    w=w,
                    covariates_scaling=covariates_scaling,
                    append_original=False) # n_aug x w
                num_augmentations = len(labels)
                assert num_augmentations == w_augmented.shape[0]
                identities = [identity] # n_id = 1
                w_augmented = w_augmented.unsqueeze(0) # n_id x n_aug x dim_w
                c = reference_sample.c_label # 1 x c | None
                if c is not None:
                    c = einops.repeat(c, '1 c -> 1 n c', n = num_augmentations)
                self.save_to_collection(
                    collection=self.sample_collection,
                    identities=identities,
                    latents=w_augmented,
                    camera=c,
                    label=labels)
                if self.generate_images:
                    self.save_images(
                        identities=identities,
                        latents=w_augmented,
                        cameras=c,
                        label=labels)
        self.sample_collection.save()

    # ---

    def augment_identities(
            self,
            algorithm : str,
            identities_references : str | None,
            covariates_scaling : LatentEdit.CovariantScaling,
            dispersion_parameters : DispersionParameters
            ) -> None:
        """ Create all augmentations for the identities. """
        if identities_references is not None:
            assert os.path.normpath(identities_references) != \
                    os.path.normpath(self.root_directory) and \
                    os.path.normpath(identities_references) != \
                    os.path.normpath(os.path.join(self.root_directory, 'samples.h5'))
            click.echo('Copying references...')
            if os.path.isdir(identities_references):
                identities_references = os.path.join(identities_references, 'samples.h5')
            references_collection = SampleCollection(
                file_path=identities_references,
                read_only=True)
            references_collection.load(
                device=self.device,
                dtype=self.dtype,
                identities=self.identities)
            for identity in self.identities:
                labels = references_collection.list_identity_labels(identity=identity)
                assert len(labels) == 1, 'Reference collection contain augmentations'
                assert labels[0] == 'reference'
                sample = references_collection.get_sample(
                    identity=identity,
                    label='reference')
                self.sample_collection.add_sample(
                    identity=identity,
                    sample=sample,
                    label='reference')
        if algorithm == 'dispersion':
            self.augment_identities_dispersion(
                dispersion_parameters=dispersion_parameters,
                covariates_scaling=covariates_scaling)
        elif algorithm == 'covariates':
            self.augment_identities_directions(
                covariates_scaling=covariates_scaling)
        else:
            raise Exception(f'Unknown algorithm : {algorithm}')

# ---

@click.group(chain=True, help='Generate a synthetic database')
@click.pass_context
@click.option('--network-type', '-nt', help='Network type', type=click.Choice(utils.network_types()), default='stylegan2')
@click.option('--network-path', '-nt', help='Path to network weights', default=None)
@click.option('--embedding-type', '-et', help='Embedding type', type=click.Choice(Embedding.get_available_models()), default='iresnet50')
@click.option('--num-identities', '-n', default=100, type=int, help='Number of identities in the database')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--root-directory', '-r', type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True), required=True,
              help='Root of the output directory tree')
@click.option('--postprocessor-config', '-pc', type=click.Choice(Cropper.get_output_configs()), default=None, help='Post-processor cropper config')
@click.option('--seed', '-s', type=int, default=None, help='Random seed. Fix for reproducibility.')
@click.option('--batch-size', '-b', type=int, default=DatabaseGenerator.ParallelParameters.batch_size, help='Inference batch size.')
@click.option('--disable-progress-bar', is_flag=True, help='Do not show progress bar (for file output)')
@click.option('--no-images', is_flag=True, help='Do not generate images')
def generate_database(
        ctx : click.Context,
        network_type : str,
        network_path: str | None,
        embedding_type : str,
        num_identities : int,
        truncation_psi : float,
        root_directory : str,
        postprocessor_config : str | None,
        seed : int | None,
        batch_size : int,
        disable_progress_bar : bool,
        no_images: bool
        ) -> None:
    current_task, num_tasks = utils.get_task()
    parallel_parameters = DatabaseGenerator.ParallelParameters(
        batch_size=batch_size,
        job_parallel= True if num_tasks > 1 else False,
        job_num=num_tasks,
        job_rank=current_task)
    identities = list(range(num_identities))[current_task::num_tasks]
    database_generator = DatabaseGenerator(
        identities=identities,
        network_type=network_type,
        network_path=network_path,
        embedding_type=embedding_type,
        truncation_psi=truncation_psi,
        parallel_parameters=parallel_parameters,
        root_directory=root_directory,
        postprocessor_config=postprocessor_config,
        disable_progress_bar=disable_progress_bar,
        generate_images=not no_images,
        seed=seed)
    ctx.obj = database_generator
    utils.dump_command(ctx, path=database_generator.root_directory)

# ---

@generate_database.command(help='Create identities references')
@click.pass_context
@click.option('--algorithm', '-a', help='Algorithm to use', type=click.Choice(DatabaseGenerator.REFERENCES_ALGORITHMS), default='random')
@click.option('--ict', '-i', type=float, default=DatabaseGenerator.ReferencesParameters.ict, help='Minimum interclass distance between distinct identities')
@click.option('--repulse', '-r', type=float, default=DatabaseGenerator.ReferencesParameters.repulse, help='Repulsion force between identities')
@click.option('--neutralization', is_flag=True, help='Do neutralize references')
@click.option('--langevin-iterations', type=int, help='Number of iterations for the Langevin algorithm',
              default=DatabaseGenerator.LangevinParameters.iterations, show_default=True)
@click.option('--langevin-checkpoint-iter', type=int, help='Number of iterations between checkpoints, no checkpoints if zero or negative.',
              default=DatabaseGenerator.LangevinParameters.checkpoint_iter, show_default=True)
@click.option('--langevin-repulsion-coefficient', '-k0', 'langevin_repulsion_coefficient', type=float, help='Repulsion coefficient for a pair of identities',
              default=DatabaseGenerator.LangevinParameters.repulsion_coefficient, show_default=True)
@click.option('--langevin-repulsion-radius', '-r0', 'langevin_repulsion_radius', type=float, help='Maximum distance of the repulsion interaction',
              default=DatabaseGenerator.LangevinParameters.repulsion_radius, show_default=True)
@click.option('--langevin-timestep', '-dt', 'langevin_timestep', type=float,
              help='Timestep for the Langevin algorithm, for negative values the timestep is calculated automatically',
              default=DatabaseGenerator.LangevinParameters.timestep, show_default=True)
@click.option('--langevin-viscous-coefficient', '-gamma', 'langevin_viscous_coefficient', type=float, help='',
              default=DatabaseGenerator.LangevinParameters.viscous_coefficient, show_default=True)
@click.option('--langevin-random-force', '-eta0', 'langevin_random_force', type=float, help='',
              default=DatabaseGenerator.LangevinParameters.random_force, show_default=True)
@click.option('--langevin-latent-force', '-kl', 'langevin_latent_force', type=float, help='',
              default=DatabaseGenerator.LangevinParameters.latent_force, show_default=True)
@click.option('--langevin-constant-embeddings', default=None, type=click.Path(file_okay=True, dir_okay=False), help='Load constant embeddings file')
@click.option('--langevin-constant-distance', '-cd', 'langevin_constant_distance', type=float, help='',
              default=DatabaseGenerator.LangevinParameters.constant_distance, show_default=True)
@click.option('--camera-fov', 'camera_fov', type=float, help='Camera field of view (EG3D)', default=20.0, show_default=True)
def create_references(
        ctx : click.Context,
        algorithm : str,
        ict : float,
        repulse : float,
        neutralization : bool,
        langevin_iterations : int,
        langevin_checkpoint_iter : int,
        langevin_repulsion_coefficient : float,
        langevin_repulsion_radius : float,
        langevin_timestep : float,
        langevin_viscous_coefficient : float,
        langevin_random_force : float,
        langevin_latent_force : float,
        langevin_constant_embeddings : str,
        langevin_constant_distance : float,
        camera_fov : float
        ):
    database_generator : DatabaseGenerator = ctx.obj
    utils.dump_command(ctx, path=database_generator.root_directory, subcommand=True)
    references_parameters = DatabaseGenerator.ReferencesParameters(
        ict=ict,
        repulse=repulse,
        neutralisation=neutralization,
        camera_fov=camera_fov)
    langevin_parameters = DatabaseGenerator.LangevinParameters(
        iterations=langevin_iterations,
        repulsion_coefficient=langevin_repulsion_coefficient,
        repulsion_radius=langevin_repulsion_radius,
        timestep=langevin_timestep,
        viscous_coefficient=langevin_viscous_coefficient,
        random_force=langevin_random_force,
        latent_force=langevin_latent_force,
        checkpoint_iter=langevin_checkpoint_iter,
        constant_embeddings_path=langevin_constant_embeddings,
        constant_distance=langevin_constant_distance)
    database_generator.create_references(
        algorithm=algorithm,
        references_parameters=references_parameters,
        langevin_parameters=langevin_parameters)
# ---

@generate_database.command(help='Create variations of the synthetic identities')
@click.pass_context
@click.option('--algorithm', '-a', help='Algorithm to use', type=click.Choice(DatabaseGenerator.VARIATIONS_ALGORITHMS), default='dispersion')
@click.option('--dispersion-iterations', type=int, help='', default=DatabaseGenerator.DispersionParameters.iterations, show_default=True)
@click.option('--dispersion-checkpoint-iter', type=int, help='Number of iterations between checkpoints, no checkpoints if zero or negative.',
              default=DatabaseGenerator.DispersionParameters.checkpoint_iter, show_default=True)
@click.option('--dispersion-num-augmentations', type=int, help='', default=DatabaseGenerator.DispersionParameters.num_augmentations, show_default=True)
@click.option('--dispersion-embedding-coefficient', type=float, help='', default=DatabaseGenerator.DispersionParameters.embedding_coefficient, show_default=True)
@click.option('--dispersion-latent-coefficient', type=float, help='', default=DatabaseGenerator.DispersionParameters.latent_coefficient, show_default=True)
@click.option('--dispersion-latent-radius', type=float, help='', default=DatabaseGenerator.DispersionParameters.latent_radius, show_default=True)
@click.option('--dispersion-latent-global-coefficient', type=float, help='', default=DatabaseGenerator.DispersionParameters.latent_global_coefficient, show_default=True)
@click.option('--dispersion-random-force', type=float, help='', default=DatabaseGenerator.DispersionParameters.random_force, show_default=True)
@click.option('--dispersion-viscous-coefficient', type=float, help='', default=DatabaseGenerator.DispersionParameters.viscous_coefficient, show_default=True)
@click.option('--dispersion-timestep', type=float, help='', default=DatabaseGenerator.DispersionParameters.timestep, show_default=True)
@click.option('--dispersion-initial-noise', type=float, default=DatabaseGenerator.DispersionParameters.initial_noise, help='Strength of the initialization noise')
@click.option('--dispersion-initial-covariates', is_flag=True, help='Activate covariates initialization')
@click.option('--dispersion-camera-fov', type=float, help='', default=DatabaseGenerator.DispersionParameters.camera_fov, show_default=True)
@click.option('--dispersion-camera-angle-variation', type=float, help='', default=DatabaseGenerator.DispersionParameters.camera_angle_variation, show_default=True)
@click.option('--pose-scaling', '-ps', type=float, default=LatentEdit.CovariantScaling.pose, help='Strength of the pose augmentation')
@click.option('--illumination-scaling', '-is', type=float, default=LatentEdit.CovariantScaling.illumination, help='Strength of the illumination augmentation')
@click.option('--expression-scaling', '-es', type=float, default=LatentEdit.CovariantScaling.expression, help='Strength of the expression augmentation')
@click.option('--identities-references', '-ir', type=click.Path(exists=True, file_okay=True, dir_okay=True), default=None,
              help= 'Path to the sample collection containing identities references to augment.'
                    'If the path is a directory, the sample.h5 collection is loaded.'
                    'The default sample collection in the root directory is used if unspecified.')
def create_variations(
        ctx : click.Context,
        algorithm : str,
        dispersion_iterations : int,
        dispersion_checkpoint_iter : int,
        dispersion_num_augmentations : int,
        dispersion_embedding_coefficient : float,
        dispersion_latent_coefficient : float,
        dispersion_latent_radius : float,
        dispersion_latent_global_coefficient : float,
        dispersion_random_force : float,
        dispersion_viscous_coefficient : float,
        dispersion_timestep : float,
        dispersion_initial_noise : float,
        dispersion_initial_covariates : bool,
        dispersion_camera_fov : float,
        dispersion_camera_angle_variation : float,
        pose_scaling : float,
        illumination_scaling : float,
        expression_scaling : float,
        identities_references : str
        ) -> None:
    database_generator : DatabaseGenerator = ctx.obj
    utils.dump_command(ctx, path=database_generator.root_directory, subcommand=True)
    covariates_scaling = LatentEdit.CovariantScaling(
        pose=pose_scaling,
        illumination=illumination_scaling,
        expression=expression_scaling)
    dispersion_parameters = DatabaseGenerator.DispersionParameters(
        iterations=dispersion_iterations,
        checkpoint_iter=dispersion_checkpoint_iter,
        num_augmentations=dispersion_num_augmentations,
        embedding_coefficient=dispersion_embedding_coefficient,
        latent_coefficient=dispersion_latent_coefficient,
        latent_radius=dispersion_latent_radius,
        latent_global_coefficient=dispersion_latent_global_coefficient,
        random_force=dispersion_random_force,
        viscous_coefficient=dispersion_viscous_coefficient,
        timestep=dispersion_timestep,
        initial_noise=dispersion_initial_noise,
        initial_covariates=dispersion_initial_covariates,
        camera_fov=dispersion_camera_fov,
        camera_angle_variation=dispersion_camera_angle_variation)
    if database_generator.parallel_parameters.job_parallel == False or \
            database_generator.parallel_parameters.job_rank == 0:
        command_parameters = [{ctx.info_name : ctx.params}]
        with open(os.path.join(database_generator.root_directory, 'command.yml'), 'a') as file:
            yaml.dump(command_parameters, file)
    database_generator.augment_identities(
        algorithm=algorithm,
        identities_references=identities_references,
        covariates_scaling=covariates_scaling,
        dispersion_parameters=dispersion_parameters)

# ---

@generate_database.command(help='Merge stats files')
@click.pass_context
@click.option('--remove', '-r', help='Remove original files', is_flag=True)
def merge_stats(
        ctx : click.Context,
        remove : bool,
        ) -> None:
    database_generator : DatabaseGenerator = ctx.obj
    utils.dump_command(ctx, path=database_generator.root_directory, subcommand=True)
    database_generator.merge_stats_files(remove=remove)

# ---

@generate_database.command(help='Merge sample collections')
@click.pass_context
@click.option('--remove', '-r', help='Remove original files', is_flag=True)
@click.option('--references-collection', '-rc', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
              help= 'Path to the sample collection containing references,'
                    'use this if the references collection is in another directory')
def merge_samples(
        ctx : click.Context,
        remove : bool,
        references_collection : str
        ) -> None:
    database_generator : DatabaseGenerator = ctx.obj
    utils.dump_command(ctx, path=database_generator.root_directory, subcommand=True)
    if references_collection is not None:
        references_collection = SampleCollection(
            file_path=references_collection,
            read_only=True)
    else:
        references_collection = None
    database_generator.merge_sample_collections(
        reference_collection=references_collection,
        remove=remove)

# ---

@generate_database.command(help='Generate images from a sample collection')
@click.pass_context
@click.option('--image-directory', '-d', type=click.Path(exists=False, file_okay=False, dir_okay=True), default=None,
              help='Path to the directory where images are to be saved')
def generate_images(
        ctx : click.Context,
        image_directory : str | None
        ) -> None:
    database_generator : DatabaseGenerator = ctx.obj
    utils.dump_command(ctx, path=database_generator.root_directory, subcommand=True)
    database_generator.generate_all_images(image_directory=image_directory)
