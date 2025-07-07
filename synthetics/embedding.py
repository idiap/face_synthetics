#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import click
from tqdm import tqdm

import torch
import einops
import kornia

from . import utils
from .utils import Sample, SampleCollection
from .cropper import Cropper

from .iresnet.iresnet import iresnet34, iresnet50, iresnet100
from .adaface import net as adaface_net
from .databases import databases, Database
from .databases import FFHQDatabase, MultipieDatabase

# ---

class Embedding():

    def __init__(
            self,
            model_name : str = 'iresnet50',
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32
            ) -> None:
        models_names= self.get_available_models()
        assert model_name in models_names, f'Unknown model type: {model_name}'
        self.model_name = model_name
        self.backbone_type = utils.models[model_name]['backbone_type']
        self.architecture = utils.models[model_name]['architecture']
        self.device = device
        self.dtype = dtype
        self.e_dim = 512
        self.input_width = 112
        self.input_height = 112
        checkpoint_path = utils.get_model_path(model_name=model_name)
        if self.backbone_type == 'iresnet':
            self.cropper_config = Cropper.Config.iresnet_bob
            if self.architecture == 'iresnet34':
                self.model = iresnet34(checkpoint_path=checkpoint_path)
            elif self.architecture == 'iresnet50':
                self.model = iresnet50(checkpoint_path=checkpoint_path)
            elif self.architecture == 'iresnet100':
                self.model = iresnet100(checkpoint_path=checkpoint_path)
            else:
                raise RuntimeError(f'Unknown architecture {self.architecture}')
        elif self.backbone_type == 'adaface':
            self.cropper_config = Cropper.Config.arcface
            self.model = adaface_net.build_model(self.architecture)
            statedict = torch.load(checkpoint_path)['state_dict']
            model_statedict = \
                {key[6:]:val \
                    for key, val in statedict.items() \
                        if key.startswith('model.')}
            self.model.load_state_dict(model_statedict)
        else:
            raise RuntimeError(f'Unsupported backbone type {self.backbone_type}')
        self.model.eval()
        self.model.to(self.device)

    # ---

    @staticmethod
    def get_available_models() -> list[str]:
        model_names = utils.models.keys()
        face_models = [model_name for model_name in model_names \
                            if utils.models[model_name]['type'] == 'embedding']
        return face_models

    # ---

    def extract(
            self,
            image : torch.Tensor
            ) -> torch.Tensor:
        """ Extracts the face embedding from an image """
        assert type(image) == torch.Tensor
        assert image.ndim == 4
        assert image.shape[1] == 3
        assert image.shape[2] == self.input_height
        assert image.shape[3] == self.input_width
        if self.backbone_type == 'iresnet':
            return self.model(image)
        elif self.backbone_type == 'adaface':
            # b x 3 x h x w [-1.0, 1.0] BGR
            image_bgr = kornia.color.rgb_to_bgr(image)
            return self.model(image_bgr)[0]
        else:
            raise RuntimeError('Unknown backbone')

    # ---

    def distance(
            self,
            e1 : torch.Tensor,
            e2 : torch.Tensor
            ) -> torch.Tensor:
        """
            Compute angular distance between two (batches of) embeddings.
            Vectors are normalized, i.e. distances are on the unit hyper-sphere.
        """
        assert isinstance(e1, torch.Tensor)
        assert e1.ndim in [1,2]
        if e1.ndim == 1:
            e1 = e1.unsqueeze(0)
        assert e1.shape[1] == self.e_dim
        assert isinstance(e2, torch.Tensor)
        assert e2.ndim in [1,2]
        if e2.ndim == 1:
            e2 = e2.unsqueeze(0)
        assert e2.shape[1] == self.e_dim
        e1_norm = torch.linalg.vector_norm(e1, dim=1)
        e2_norm = torch.linalg.vector_norm(e2, dim=1)
        e1_norm = einops.repeat(e1_norm, 'b -> b d', d=self.e_dim)
        e2_norm = einops.repeat(e2_norm, 'b -> b d', d=self.e_dim)
        e1 = e1 / e1_norm
        e2 = e2 / e2_norm
        dot = torch.einsum('ai,bi->ab', e1, e2) # TODO matmul might be a little faster
        # TODO: TEMP factor avoid NaNs
        angle = torch.arccos(dot * .999999)
        if torch.isnan(angle).any():
            raise RuntimeError(f'Found NaN: {e1.shape} {e2.shape} {e1_norm} {e2_norm}')
        return angle

# ---

@click.command(
    help='Extracts the embedding from an image')
@click.pass_context
@click.option(
    '--model', 
    '-m', 
    help='Model name', 
    type=click.Choice(Embedding.get_available_models()), 
    default='iresnet50')
@click.option(
    '--input', 
    '-i', 
    type=str, 
    help='Input image file', 
    default=None)
@click.option(
    '--out', 
    '-o', 
    type=click.Path(file_okay=True, dir_okay=False), 
    help='File path to save the result', 
    default='out.h5', 
    show_default=True)
@click.option(
    '--device', 
    '-d', 
    type=click.Choice(['cpu', 'cuda']), 
    help='Device to perform the calculation', 
    default='cpu', 
    show_default=True)
def get_embedding(
        ctx : click.Context,
        model : str,
        input : str,
        out: str,
        device : str):
    device = torch.device(device=device)
    cropper = Cropper(
        input_config=Cropper.Config.dlib,
        output_config=Cropper.Config.iresnet_bob,
        device=device)
    embedding = Embedding(model_name=model, device=device)
    image = utils.load_image(file_path=input, device=device)
    image = cropper.crop(image)
    emb = embedding.extract(image=image)
    sample = Sample(embedding_type=model, embedding=emb)
    sample.save(file_path=out, create_directories=True)

# ---

@click.command(
    help='Extracts embeddings from a database')
@click.pass_context
@click.option(
    '--model', 
    help='Model name', 
    type=click.Choice(Embedding.get_available_models()), 
    default='iresnet50')
@click.option(
    '--database', 
    type=click.Choice(databases), 
    help='Input database', 
    required=True)
@click.option(
    '--output', 
    type=click.Path(file_okay=True, dir_okay=False), 
    help='File path to save the result', 
    required=True)
@click.option(
    '--batch-size', 
    type=int, 
    help='Batch size', 
    default=8)
@click.option(
    '--device', 
    type=click.Choice(['cpu', 'cuda']), 
    help='Device to perform the calculation', 
    default='cpu', 
    show_default=True)
def get_database_embeddings(
        ctx : click.Context,
        model : str,
        database : str,
        output: str,
        batch_size : int,
        device : str):
    device = torch.device(device=device)
    cropper = Cropper(
        input_config=Cropper.Config.dlib,
        output_config=Cropper.Config.iresnet_bob,
        device=device)
    crop_colors = 3
    crop_height = Cropper.Config.iresnet_bob.value.height
    crop_width = Cropper.Config.iresnet_bob.value.width
    embedding = Embedding(model_name=model, device=device)
    sample_collection = SampleCollection(file_path=output)
    protocol_names = None
    group_names = None
    if database == 'multipie':
        db = MultipieDatabase()
    elif database == 'ffhq':
        db = FFHQDatabase()
    else:
        raise RuntimeError('Unknown database')
    click.echo(f'Loading database ...')
    database_samples : list[Database.Sample] = db.query(protocol_names=protocol_names, group_names=group_names)
    num_samples = len(database_samples)
    click.echo(f'Loaded {num_samples} samples from database ...')
    batch_shape = (batch_size, crop_colors, crop_height, crop_width)
    image_batch = torch.zeros(batch_shape, device=device, dtype=torch.float32)
    for i0 in tqdm(range(0, num_samples, batch_size)):
        i1 = min(i0 + batch_size, num_samples)
        image_success = torch.zeros((batch_size, ), device=torch.device('cpu'), dtype=torch.bool)
        for i in range(i0, i1):
            try:
                database_sample = database_samples[i]
                image = db.load_sample(sample=database_sample, device=device)
                image_cropped = cropper.crop(image)
                image_batch[i - i0, :, :, :] = image_cropped[0, :, :, :]
                image_success[i - i0] = True
            except Exception as e:
                image_success[i - i0] = False
                click.echo(f'Error while processing file: {str(e)}')
        embedding_batch = embedding.extract(image=image_batch)
        embedding_batch = embedding_batch.detach()
        for i in range(i0, i1):
            if image_success[i - i0]:
                database_sample = database_samples[i]
                e = embedding_batch[i - i0, :].unsqueeze(0)
                sample = Sample(embedding_type=model, embedding=e)
                sample_collection.add_sample(
                    identity=database_sample.identity,
                    label=database_sample.key,
                    sample=sample)
        sample_collection.save() # TEMP
    sample_collection.save()
