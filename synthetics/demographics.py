#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from typing import TYPE_CHECKING
from pathlib import Path
import click
import torch as pt
import numpy as np
import synthetics.utils as su
import synthetics.tools.instantiate as sti
import synthetics.tools.batches as stb
import synthetics.generator as sg


if TYPE_CHECKING:
    from synthetics.densities.compressed import CompressedGMMs


def _ensure_extension(filename: Path, ext: str) -> Path:
    if filename.suffix != ext:
        return filename.parent / (filename.stem + ext)
    return filename


@click.command(
    help='Generate images by sampling from a demographic-based density model')
@click.option(
    '--density-type',
    '-dt',
    help='Configuration of the density model',
    type=click.Choice(su.density_model_types()))
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
    '--output_folder',
    '-o',
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help='Location where to place the generate samples')
@click.option(
    '--batch-size',
    type=int,
    help='Number if images in a single batch',
    default=16,
    show_default=True)
def generate_demographics(
    density_type: str,
    seed : int,
    number : int,
    output_folder: str | Path | None,
    batch_size: int,
) -> None:
    """Generate images by sampling from a demographic-based density model"""

    # Seed
    # ---------------------
    if seed is not None:
        pt.manual_seed(seed)
        np.random.seed(seed)

    # Output
    # ---------------------
    if output_folder is None:
        output_folder = Path.cwd()
    elif isinstance(output_folder, str):
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Density model
    # ---------------------
    cfg = su.models[density_type]
    density: "CompressedGMMs" = sti.instantiate(cfg["net"])

    # Generator
    # ---------------------
    generator = sg.Generator(network_type=cfg["network_type"])

    # Sample + Synthetize
    # ---------------------
    _images = []
    ws, labels = density(number, batch_size)
    for w in stb.make_batch(ws, batch_size):
        img = generator.synthesis(w=w, w_plus=w.ndim == 3)
        _images.append(img.cpu())
    images = pt.concat(_images)

    # Save
    # ---------------------
    # Images
    label_filename = output_folder / "labels.txt"
    with label_filename.open("wt") as f:
        f.write("Demographic-label\n")
        for k, (image, label) in enumerate(zip(images, labels)):
            # Image
            filename = output_folder / f"images/{k:05d}" / "reference.jpg"
            filename.parent.mkdir(parents=True, exist_ok=True)
            su.save_image(image.unsqueeze(0), file_path=filename.as_posix())
            # Label
            f.write(f"{label}\n")

    # Samples
    h5_filename = _ensure_extension(output_folder / "reference", ext=".h5")
    samples = su.SampleCollection(h5_filename.as_posix())
    for k, w in enumerate(ws):
        _w = w.unsqueeze(0).cpu()   # Add batch dim again
        w_key = "w_plus_latent" if  _w.ndim > 2 else "w_latent"
        kw = {"network_type": cfg["network_type"],
              w_key: _w}
        sample = su.Sample(**kw)
        samples.add_sample(identity=k, label="reference", sample=sample)
    samples.save()
