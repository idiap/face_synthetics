#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import os
import sys
import shutil
import urllib.request
import bz2
import yaml
import click
import subprocess
from dataclasses import dataclass

import torch
import torchvision.utils
import einops
import numpy as np
import h5py
from PIL import Image
import imageio
import gdown

# ---

def __parse_config_file():
    path_usr = os.path.expanduser('~')
    path_usr = os.path.join(path_usr, '.synthetics.yml')
    path_cur = './synthetics.yml'
    # try loading the file in user's directory first
    if os.path.exists(path_usr):
        path = path_usr
    elif os.path.exists(path_cur):
        path = path_cur
    else:
        raise RuntimeError('Cannot find configuration file ...')
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        return data

# ---

__config_data = __parse_config_file()
config = __config_data['config']
source_path = config['source']
models: dict[str, dict[str, str]] = __config_data['models']

# ---

def get_model_path(model_name : str) -> str:
    """ Get absolute path to a given model. The model is downloaded if needed. """
    assert model_name in models.keys()
    local_path = os.path.join(config['models'], models[model_name]['file_name'])
    if not os.path.exists(local_path):
        click.echo(f'downloading model: {model_name} ...')
        # check is exists on local network
        if 'local_model_zoo' in models[model_name].keys() and \
            os.path.exists(models[model_name]['local_model_zoo']):
            click.echo('... getting local model zoo version')
            shutil.copy2(models[model_name]['local_model_zoo'], local_path)
        elif 'web_url' in models[model_name].keys():
            model_url = models[model_name]['web_url']
            click.echo(f'... getting web version: {model_url}')
            if 'compression' in models[model_name].keys():
                if models[model_name]['compression'] == 'bz2':
                    click.echo('... decompressing')
                    tmp_file, _ = urllib.request.urlretrieve(models[model_name]['web_url'])
                    with bz2.BZ2File(tmp_file, 'rb') as src, open(local_path, 'wb') as dst:
                        dst.write(src.read())
                else:
                    raise RuntimeError('unknown compression format')
            else:
                urllib.request.urlretrieve(models[model_name]['web_url'], local_path)
        elif 'gdrive_id' in models[model_name].keys():
            model_id = models[model_name]['gdrive_id']
            click.echo(f'... getting gdrive version: {model_id}')
            gdown.download(
                output=local_path, id=model_id, use_cookies=False, fuzzy=True)
        else:
            raise RuntimeError('cannot find model :(')
        click.echo('... done!')
    return local_path

# ---

def network_types():
    """ Returns generative networks names. """
    network_types = []
    for net in models.keys():
        if ('type' in models[net].keys() and
                models[net]['type'].startswith('gan')):
            network_types.append(net)
    return network_types


def nvidia_network_types() -> list[str]:
    """List of official network arch"""
    arch_types = []
    for name, options in models.items():
        if 'type' in options and options['type'] == "gan":
            arch_types.append(name)
    return arch_types


def _external_types(arch: str) -> list[str]:
    arch_types = []
    for name, options in models.items():
        if 'type' in options and options['type'] == arch:
            arch_types.append(name)
    return arch_types

def lucidrains_network_types() -> list[str]:
    """
    List of lucidrains network arch (i.e, reimplementation of nvidia network)
    """
    return _external_types(arch="gan-lucidrains")


def rosinality_network_types() -> list[str]:
    """
    List of rosinality network arch (i.e, reimplementation of nvidia network)
    """
    return _external_types(arch="gan-rosinality")


# ---

# pyright: reportMissingImports=false
def get_network(
        network_type : str,
        network_path: str | None = None,
        require_grad : bool = False,
        device : torch.device = torch.device('cuda')) -> object:
    """ Loads a generative network. """
    if network_type == 'eg3d':
        sys.path.insert(0, os.path.join(config['source'], 'synthetics/eg3d'))
        import dnnlib
        import legacy
        fp = dnnlib.util.open_url(get_model_path(network_type))
        G = legacy.load_network_pkl(fp)['G_ema']
        G = G.requires_grad_(require_grad).to(device)
        return G
    elif network_type in nvidia_network_types():
        sys.path.insert(0, os.path.join(config['source'], 'synthetics/stylegan3'))
        import dnnlib
        import legacy
        fp = dnnlib.util.open_url(get_model_path(network_type))
        G = legacy.load_network_pkl(fp)['G_ema']
        G = G.requires_grad_(require_grad).to(device)
        return G
    elif network_type in lucidrains_network_types():
        import synthetics.adapters.lucidrains as al
        # Naming: <arch>-lucidrains-<resolution>
        image_size = int(network_type.split("-")[-1])
        # Adaptater model
        gen = al.GeneratorAdapter(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=image_size,
            img_channels=3,
            synthesis_kwargs={"remap_output": True})
        # Reload
        if network_path is None:
            network_path = get_model_path(network_type)
        state_dict = torch.load(
            network_path,
            map_location="cpu",
            weights_only=True)
        gen.load_state_dict(state_dict)
        return gen.requires_grad_(require_grad).to(device)
    elif network_type in rosinality_network_types():
        import synthetics.adapters.rosinality as ar
        # Naming: <arch>-rosinality-<resolution>
        image_size = int(network_type.split("-")[-1])
        # Adaptater model
        gen = ar.GeneratorAdapter(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=image_size,
            img_channels=3)
        # Reload
        if network_path is None:
            network_path = get_model_path(network_type)
        state_dict = torch.load(
            network_path,
            map_location="cpu",
            weights_only=True)
        gen.load_state_dict(state_dict)
        return gen.requires_grad_(require_grad).to(device)
    else:
        raise RuntimeError('Unknown network type')

# ---

def get_database_index_path(file_name : str) -> str:
    """ Get the path to a database index, download it if needed. """
    assert 'databases' in config
    assert 'user_index_directory' in config['databases']
    assert 'local_index_repository' in config['databases']
    index_directory = config['databases']['user_index_directory']
    local_repository = config['databases']['local_index_repository']
    file_path = os.path.join(index_directory, file_name)
    if not os.path.exists(file_path):
        # TODO web download
        local_file_path = os.path.join(local_repository, file_name)
        click.echo(f'Downloading database index: {file_name} ...')
        if not os.path.exists(local_file_path):
            raise RuntimeError(f'Cannot find {local_file_path}')
        click.echo(f'... getting local version: {local_file_path}')
        shutil.copy2(local_file_path, file_path)
        click.echo('... done!')
    else:
        click.echo(f'Using database index: {file_path} ...')
    return file_path

# ---

def get_database_data_directory(database_name : str) -> str:
    assert 'databases' in config
    assert database_name in config['databases']
    assert 'data_path' in config['databases'][database_name]
    return config['databases'][database_name]['data_path']

# ---

def adjust_dynamic_range(
        image : torch.Tensor,
        input_range_min : float | torch.Tensor = None,
        input_range_max : float | torch.Tensor = None,
        target_range_min : float = -1.0,
        target_range_max : float = 1.0,
        clamp : bool = True
        ) -> torch.Tensor:
    """
        Adjust the dynamic range of an image tensor or batch thereof.
        Data format (N, C, H, W). Default target range [-1.0, 1.0].
        In batch mode (N>1) the input_range_min and input_range_max
        parameters can be 1D torch.Tensor for per sample processing.
    """
    assert type(image) == torch.Tensor, 'Image must be a torch.Tensor object.'
    assert image.ndim == 4, 'Image must be 4D (N, C, H, W).'
    assert type(target_range_min) == float, 'Target range must be float'
    assert type(target_range_max) == float, 'Target range must be float'
    batch_size = image.shape[0]
    device = image.device
    dtype = image.dtype
    if input_range_min is None:
        if batch_size > 1:
            input_range_min = torch.amin(image, dim=(1,2,3))
            input_range_min = einops.rearrange(input_range_min, 'b -> b 1 1 1')
        else:
            input_range_min = torch.min(image)
            input_range_min = einops.rearrange(input_range_min, '-> 1 1 1 1')
    elif type(input_range_min) == float:
        input_range_min = torch.tensor(input_range_min, device=device, dtype=dtype)
        input_range_min = einops.repeat(input_range_min, '-> b 1 1 1', b=batch_size)
    elif type(input_range_min) == torch.Tensor:
        assert input_range_min.ndim == 1
        assert input_range_min.shape[0] == image.shape[0]
        input_range_min = einops.rearrange(input_range_min, 'b -> b 1 1 1')
    else:
        raise RuntimeError('Incorrect input_range_min type')
    if input_range_max is None:
        if batch_size > 1:
            input_range_max = torch.amax(image, dim=(1,2,3))
            input_range_max = einops.rearrange(input_range_max, 'b -> b 1 1 1')
        else:
            input_range_max = torch.max(image)
            input_range_max = einops.rearrange(input_range_max, '-> 1 1 1 1')
    elif type(input_range_max) == float:
        input_range_max = torch.tensor(input_range_max, device=device, dtype=dtype)
        input_range_max = einops.repeat(input_range_max, '-> b 1 1 1', b=batch_size)
    elif type(input_range_max) == torch.Tensor:
        assert input_range_max.ndim == 1
        assert input_range_max.shape[0] == image.shape[0]
        input_range_max = einops.rearrange(input_range_max, 'b -> b 1 1 1')
    else:
        raise RuntimeError('Incorrect input_range_max type')
    image = target_range_min * (input_range_max - image) \
          + target_range_max * (image - input_range_min)
    image = image / (input_range_max - input_range_min)
    if clamp:
        image = image.clamp(min=target_range_min, max=target_range_max)
    return image

# convert from torch image to numpy
def image_to_numpy(
        image : torch.Tensor,
        range_min : float = -1.0,
        range_max : float = 1.0
        ) -> np.ndarray:
    """
        Convert a torch image (float, range = [-1.0, 1.0]) in 4D (N, C, H, W)
        format to numpy (uint8, range = [0, 255]) format. The default input
        range can be overridden with the parameters ``range_min`` and ``range_max``.
        If ``normalize`` is set to True, the input dynamic range is calculated from
        data minimum and maximum.
    """
    assert type(image) == torch.Tensor, 'image must be a torch.Tensor object.'
    assert image.ndim == 4, 'image must be 4D (N, C, H, W).'
    image = adjust_dynamic_range(
                image=image,
                input_range_min=range_min,
                input_range_max=range_max,
                target_range_min=0.0,
                target_range_max=255.0)
    return image.detach().cpu().numpy().astype('uint8')

# numpy image to torch float image
def numpy_to_image(
        image : np.ndarray,
        device : torch.device = torch.device('cpu'),
        dtype : torch.dtype = torch.float32
        ) -> torch.Tensor:
    """
        Convert a numpy image (uint8, range = [0, 255])
        to torch (float, range = [-1.0, 1.0]) format.
        Image must be 4D (N, C, H, W). By default the image
        is loaded in the CPU memory in float32, this can be
        overridden using the device and dtype parameters.
    """
    assert type(image) == np.ndarray
    assert image.dtype == np.uint8
    image = torch.tensor(image, device=device, dtype=dtype)
    image = adjust_dynamic_range(
                image=image,
                input_range_min=0.0,
                input_range_max=255.0,
                target_range_min=-1.0,
                target_range_max=1.0)
    return image

# adapted from bob.io.base
def numpy_to_matplotlib(
        image : np.ndarray
        ) -> np.ndarray:
    """
        Returns a view of the image from internal format to matplotlib format.
        This function works with images, batches of images, videos, and higher
        dimensional arrays that contain images.
    """
    assert type(image) == np.ndarray
    assert image.dtype == np.uint8
    if image.ndim < 3:
        return image
    return np.moveaxis(image, -3, -1)

# adapted from bob.io.base
def matplotlib_to_numpy(
        image : np.ndarray
        ) -> np.ndarray:
    """
        Returns a view of the image from matplotlib format to Bob format.
        This function works with images, batches of images, videos, and
        higher dimensional arrays that contain images.
    """
    assert type(image) == np.ndarray
    assert image.dtype == np.uint8
    if image.ndim < 3:
        return image
    return np.moveaxis(image, -1, -3)

# adapted from bob.io.base
def numpy_to_pil(
        image : np.ndarray
        ) -> Image:
    """
        Converts a numpy uint8 image to a Pillow Image.
    """
    assert type(image) == np.ndarray
    assert image.dtype == np.uint8
    image = numpy_to_matplotlib(image)
    return Image.fromarray(image)

# adapted from bob.io.base
def pil_to_numpy(
        image : Image
        ) -> np.ndarray:
    """
        Converts an RGB or gray-scale PIL image to uint8 numpy format.
    """
    assert type(image) == Image
    return matplotlib_to_numpy(np.array(image))


# adapted from bob.io.base
def opencvbgr_to_numpy(
        image : np.ndarray
        ) -> np.ndarray:
    """
        Returns a view of the image from OpenCV BGR format to Bob RGB format.
        This function works with images, batches of images, videos, and higher
        dimensional arrays that contain images.
    """
    assert type(image) == np.ndarray
    assert image.dtype == np.uint8
    if img.ndim < 3:
        return img
    img = img[..., ::-1]
    return matplotlib_to_numpy(img)

# adapted from bob.io.base
def numpy_to_opencvbgr(
        image : np.ndarray
        ) -> np.ndarray:
    """
        Returns a view of the image from Bob format to OpenCV BGR format. This
        function works with images, batches of images, videos, and higher
        dimensional arrays that contain images.
    """
    assert type(image) == np.ndarray
    assert image.dtype == np.uint8
    if img.ndim < 3:
        return img
    img = img[..., ::-1, :, :]
    return numpy_to_matplotlib(img)

# adapted from bob.io.base
def create_directories_safe(
        directory : str
        ) -> None:
    """
        Creates a directory if it does not exists, with concurrent access
        support. This function will also create any parent directories that might
        be required.
    """
    if directory.strip() != '':
        os.makedirs(directory, exist_ok=True)

# save image, some parts adapted from bob.io.base
def save_image(
        image : torch.Tensor,
        file_path : str,
        range_min : float = -1.0,
        range_max : float = 1.0,
        normalize : bool = False,
        num_rows : int = 8,
        create_directories : bool = False
        ) -> None:
    """
        Save a tensor to an image file, pointed by file_path. The image should
        be a 4D floating point torch.Tensor in the format (N, C, H, W). The
        default dynamic range is [-1.0, 1.0], it can be overridden with the
        parameters range_min and range_max. If the normalize flag is set to True,
        the dynamic range is adjusted according to the minimum and maximum values
        of the tensor. Batches are saved as a grid of width ``num_rows``.
    """
    assert type(image) == torch.Tensor, 'image must be a torch.Tensor object.'
    assert image.ndim == 4, 'image must be 4D (N, C, H, W).'
    if image.shape[1] == 3:
        pass # RGB
    elif image.shape[1] == 1:
        # TODO colormap
        image = einops.repeat(image, 'b 1 h w -> b c h w', c=3)
    else:
        raise RuntimeError(
            'Image should be 4D (N, C, H, W) and C=3 (RGB) or C=2 (grayscale)')

    if create_directories:
        create_directories_safe(os.path.dirname(file_path))
    if normalize:
        image = adjust_dynamic_range(image=image)
    elif range_min != -1.0 or range_max != 1.0:
        image = adjust_dynamic_range(
                    image=image,
                    input_range_min=range_min,
                    input_range_max=range_max)

    # make grid for batch
    if image.shape[0] > 1:
        image = torchvision.utils.make_grid(
            tensor=image,
            nrow=num_rows,
            padding=0,
            pad_value=0,
            normalize=False)
        image = image.unsqueeze(0)
    image = image_to_numpy(image)
    image = image[0]
    image = numpy_to_matplotlib(image)
    imageio.imwrite(
        uri=file_path,
        im=image,
        format="pillow")

# load image, some parts adapted from bob.io.base
def load_image(
        file_path : str,
        device : torch.device = torch.device('cpu'),
        dtype : torch.dtype = torch.float32,
        convert_to_rgb : bool = True,
        keep_alpha : bool = False
        ) -> torch.Tensor:
    """ Loads an image file to a ``torch.Tensor``. By default the data
        is loaded in CPU memory in ``torch.float32`` (1, 3, H, W) format. """
    assert type(file_path) == str
    if not os.path.exists(file_path):
        raise RuntimeError(f"`{file_path}' does not exist!")
    try:
        image = imageio.imread(file_path)
    except:
        raise RuntimeError(f"Cannot open `{file_path}'")
    image = np.asarray(image)
    if image.ndim > 2: # color image
        extension = os.path.splitext(file_path)[1].lower()
        # remove alpha channel from png
        if extension == ".png":
            if not keep_alpha:
                image = image[:, :, 0:3]
        image = matplotlib_to_numpy(image)
    else: # grayscale image
        if convert_to_rgb:
            image = np.stack([image, image, image])
        else:
            image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = numpy_to_image(
                image=image,
                device=device,
                dtype=dtype)
    assert image.ndim == 4, 'internal error'
    assert image.shape[0] == 1, 'internal error'
    return image

# ---

SAMPLE_FILE_MAGIC = 'synthetics'
SAMPLE_COLL_MAGIC = 'synthetics_collection'
SAMPLE_FILE_MAJOR = 2
SAMPLE_FILE_MINOR = 0
DIM = 512
EMB = 512
DIM_C = 25

# ---

class Sample:
    """ Class that stores, saves and loads a sample. """

    def __init__(
            self,
            z_latent : torch.Tensor = None,
            w_latent : torch.Tensor = None,
            w_plus_latent : torch.Tensor = None,
            c_label : torch.Tensor = None,
            embedding : torch.Tensor = None,
            network_type : str = None,
            embedding_type : str = None,
            bfm_model : dict = None
            ) -> None:
        if z_latent is not None \
            or w_latent is not None \
            or w_plus_latent is not None:
            assert network_type is not None
        if network_type == 'eg3d':
            assert c_label is not None
        if embedding is not None:
            assert embedding_type is not None
        # TODO assert shape here too
        self.z_latent = z_latent
        self.w_latent = w_latent
        self.w_plus_latent = w_plus_latent
        self.c_label = c_label
        self.embedding = embedding
        self.network_type = network_type
        self.embedding_type = embedding_type
        self.bfm_model = bfm_model

    # ---

    def save(
            self,
            file_path : str,
            create_directories : bool = False
            ) -> None:
        """ Save the sample. """
        assert isinstance(file_path, str)
        if create_directories:
            create_directories_safe(os.path.dirname(file_path))
        with h5py.File(file_path, 'w') as h5_file:
            h5_file.attrs['magic'] = SAMPLE_FILE_MAGIC
            h5_file.attrs['major'] = SAMPLE_FILE_MAJOR
            h5_file.attrs['minor'] = SAMPLE_FILE_MINOR
            self.save_into(h5_object=h5_file)

    # ---

    def save_into(
            self,
            h5_object : h5py.File | h5py.Group
            ) -> None:
        """ Save the sample into an existing HDF5 file. """
        assert  isinstance(h5_object, h5py.File) or \
                isinstance(h5_object, h5py.Group)
        if self.network_type is not None:
            h5_object.attrs['network_type'] = self.network_type
        if self.embedding_type is not None:
            h5_object.attrs['embedding_type'] = self.embedding_type
        def save_dset(data : torch.Tensor, name : str, shape : tuple):
            if data is not None:
                assert len(shape) == len(data.shape)
                for i in range(len(shape)):
                    if shape[i] is not None:
                        assert shape[i] == data.shape[i]
                data = data.detach().cpu().numpy()
                h5_object.create_dataset(name=name, data=data)
        save_dset(self.z_latent, 'z_latent', (1, DIM))
        save_dset(self.w_latent, 'w_latent', (1, DIM))
        save_dset(self.w_plus_latent, 'w_plus_latent', (1, None, DIM))
        save_dset(self.c_label, 'c_label', (1, DIM_C))
        save_dset(self.embedding, 'embedding', (1, EMB))
        if self.bfm_model is not None:
            h5_object.create_group(name='bfm_model')
            save_dset(self.bfm_model['id'], 'bfm_model/id', (1, 80))
            save_dset(self.bfm_model['exp'], 'bfm_model/exp', (1, 64))
            save_dset(self.bfm_model['tex'], 'bfm_model/tex', (1, 80))
            save_dset(self.bfm_model['angle'], 'bfm_model/angle', (1, 3))
            save_dset(self.bfm_model['gamma'], 'bfm_model/gamma', (1, 27))
            save_dset(self.bfm_model['trans'], 'bfm_model/trans', (1, 3))

    # ---

    def load(
            self,
            file_path: str,
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32
            ) -> None:
        """ Loads a sample file into the Sample object. """
        with h5py.File(file_path, 'r') as h5_file:
            attr_keys = h5_file.attrs.keys()
            assert 'magic' in attr_keys
            assert h5_file.attrs['magic'] == SAMPLE_FILE_MAGIC
            assert 'major' in attr_keys
            assert h5_file.attrs['major'] == SAMPLE_FILE_MAJOR
            assert 'minor' in attr_keys
            assert h5_file.attrs['minor'] == SAMPLE_FILE_MINOR
            self.load_from(h5_object=h5_file, device=device, dtype=dtype)

    # ---

    def load_from(
            self,
            h5_object : h5py.File | h5py.Group,
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32) -> None:
        """ Loads data from an HDF5 file or group. """
        assert isinstance(h5_object, h5py.File) or isinstance(h5_object, h5py.Group)
        datasets = h5_object.keys()
        attr_keys = h5_object.attrs.keys()
        if 'network_type' in attr_keys:
            self.network_type = h5_object.attrs['network_type']
        if 'embedding_type' in attr_keys:
            self.embedding_type = h5_object.attrs['embedding_type']
        def load_dset(name, shape=None):
            assert shape is None or isinstance(shape, tuple)
            if name in datasets:
                dset = h5_object[name][:]
                dset = torch.tensor(dset, device=device, dtype=dtype)
                if shape is not None:
                    for i in range(len(shape)):
                        dim = shape[i]
                        if dim is not None:
                            assert dset.shape[i] == dim
                return dset
            else:
                return None
        self.z_latent = load_dset('z_latent', shape=(1 ,DIM))
        self.w_latent = load_dset('w_latent', shape=(1, DIM))
        self.w_plus_latent = load_dset('w_plus_latent', shape=(1, None, DIM))
        self.c_label = load_dset('c_label', shape=(1, DIM_C))
        if (self.z_latent is not None or \
            self.w_latent is not None or \
            self.w_plus_latent is not None or \
            self.c_label is not None):
            assert self.network_type is not None
        self.embedding = load_dset('embedding', shape=(1, EMB))
        if self.embedding is not None:
            assert self.embedding_type is not None
        if 'bfm_model' in h5_object and h5_object['bfm_model'].is_group():
            self.bfm_model = {}
            self.bfm_model['id'] = load_dset('/bfm_model/id', (1, 80))
            self.bfm_model['exp'] = load_dset('/bfm_model/exp', (1, 64))
            self.bfm_model['tex'] = load_dset('/bfm_model/tex', (1, 80))
            self.bfm_model['angle'] = load_dset('/bfm_model/angle', (1, 3))
            self.bfm_model['gamma'] = load_dset('/bfm_model/gamma', (1, 27))
            self.bfm_model['trans'] = load_dset('/bfm_model/trans', (1, 3))

# ---

class SampleCollection:
    """ Class that stores several samples on disk. """

    def __init__(
            self,
            file_path : str,
            read_only : bool = False,
            ) -> None:
        self.__samples = {}
        self.__is_saved = {}
        self.read_only = read_only
        self.file_path = file_path

    # ---

    def __format_identity(self, id : int) -> str:
        return f'{id:05}'

    # ---

    def list_identities(self) -> list[int]:
        """ Return the list of identities. """
        return list(self.__samples.keys())

    # ---

    def list_identity_labels(self, identity : int) -> list[str]:
        """ Return the list of labels for a given identity. """
        if identity in self.__samples:
            return list(self.__samples[identity].keys())
        else:
            return None

    # ---

    def add_sample(
            self,
            identity : int,
            label : int | str,
            sample : Sample
            ) -> None:
        """ Add a sample to the collection, if the sample already exists it is replaced. """
        assert self.read_only == False, 'Cannot add sample to read-only collection'
        assert isinstance(identity, int)
        assert isinstance(label, int) or isinstance(label, str)
        # replace / and spaces by underscores
        label = label.replace('/', '_')
        label = label.replace(' ', '_')
        if identity not in self.__samples.keys():
            self.__samples[identity] = {}
            self.__is_saved[identity] = {}
        self.__samples[identity][label] = sample
        self.__is_saved[identity][label] = False

    # ---

    def get_sample(
            self,
            identity : int,
            label : int | str,
            ) -> Sample:
        """ Get a sample from the collection. """
        assert isinstance(identity, int)
        assert isinstance(label, int) or isinstance(label, str)
        assert identity in self.__samples.keys(), \
            f'Identity {identity} not present in collecion.'
        assert label in self.__samples[identity].keys(), \
            f'Label {label} not present for identity {identity}.'
        return self.__samples[identity][label]

    # ---

    def save(self) -> None:
        """ Saved unsaved samples to the collection file. """
        assert self.read_only == False, 'Cannot save read-only collection'
        with h5py.File(self.file_path, 'a') as h5_file:
            h5_file.attrs['magic'] = SAMPLE_COLL_MAGIC
            h5_file.attrs['major'] = SAMPLE_FILE_MAJOR
            h5_file.attrs['minor'] = SAMPLE_FILE_MINOR
            for identity in self.__samples.keys():
                identity_str = self.__format_identity(identity)
                if identity_str not in h5_file.keys():
                    identity_group = h5_file.create_group(name=identity_str)
                else:
                    identity_group = h5_file[identity_str]
                for label in self.__samples[identity].keys():
                    if self.__is_saved[identity][label] == False:
                        if label not in identity_group.keys():
                            sample_group = identity_group.create_group(name=label)
                        else:
                            sample_group = identity_group[label]
                        sample = self.__samples[identity][label]
                        assert isinstance(sample, Sample)
                        sample.save_into(h5_object=sample_group)
                        self.__is_saved[identity][label] = True

    # ---

    def load(
            self,
            device : torch.device = torch.device('cuda'),
            dtype : torch.dtype = torch.float32,
            identities : list[int] | None = None
            ) -> None:
        """ Load samples from file to memory. """
        self.__samples = {}
        self.__is_saved = {}
        with h5py.File(self.file_path, 'r') as h5_file:
            attr_keys = h5_file.attrs.keys()
            assert 'magic' in attr_keys
            assert h5_file.attrs['magic'] == SAMPLE_COLL_MAGIC
            assert 'major' in attr_keys
            assert h5_file.attrs['major'] == SAMPLE_FILE_MAJOR
            assert 'minor' in attr_keys
            assert h5_file.attrs['minor'] == SAMPLE_FILE_MINOR
            for id_str, id_group in h5_file.items():
                if isinstance(id_group, h5py.Group):
                    identity = int(id_str)
                    if identities is not None:
                        if identity not in identities:
                            continue
                    self.__samples[identity] = {}
                    self.__is_saved[identity] = {}
                    for label, label_group in id_group.items():
                        if isinstance(label_group, h5py.Group):
                            sample = Sample()
                            sample.load_from(
                                h5_object=label_group,
                                device=device,
                                dtype=dtype)
                            self.__samples[identity][label] = sample
                            self.__is_saved[identity][label] = True

# ---

# TODO add slurm backend
def get_task():
    if 'SLURM_ARRAY_TASK_ID' in os.environ and os.environ['SLURM_ARRAY_TASK_ID'] != 'undefined':
        current_task = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
        num_tasks = int(os.environ['SLURM_ARRAY_TASK_MAX'])
    else:
        current_task = 0
        num_tasks = 1
    return current_task, num_tasks

# ---

@dataclass
class git_info:
    version : str
    branch : str

def get_git_info() -> git_info:
    version = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=source_path).decode('ascii').strip()
    branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=source_path).decode('ascii').strip()
    info = git_info(version=version, branch=branch)
    return info

# ---

def dump_command(
        ctx : click.Context,
        path : str = '.',
        file : str = 'command.yml',
        subcommand : bool = False
        ) -> None:
    current_task, num_tasks = get_task()
    if num_tasks == 1 or current_task == 0:
        command_parameters = {ctx.info_name : ctx.params}
        if not subcommand:
            git_info = get_git_info()
            command_parameters[ctx.info_name]['git'] = {}
            command_parameters[ctx.info_name]['git']['branch'] = git_info.branch
            command_parameters[ctx.info_name]['git']['version'] = git_info.version
        command_parameters = [command_parameters]
        with open(os.path.join(path, file), 'a') as file:
            yaml.dump(command_parameters, file)
