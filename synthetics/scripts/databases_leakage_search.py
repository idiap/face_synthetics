#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os
from dataclasses import dataclass

import click
import torch
import einops
from tqdm import tqdm

from synthetics import Cropper, Embedding, Plot, SampleCollectionData, FFHQDatabase, utils

@dataclass
class SamplePairs:
    identities_i : torch.Tensor
    identities_j : torch.Tensor
    labels_i : torch.Tensor | None
    labels_j : torch.Tensor | None
    distances : torch.Tensor
    
def find_nearest_embeddings(
        collection_data_i: SampleCollectionData,
        collection_data_j : SampleCollectionData,
        embedding : Embedding,
        num_nearest : int = 10,
        batch_size : int = 4096,
        no_identity_twice: bool = True
        ) -> SamplePairs:
    embedding_i = collection_data_i.embeddings
    embedding_j = collection_data_j.embeddings
    min_distances : torch.Tensor | None = None
    min_index_i : torch.Tensor | None = None
    min_index_j : torch.Tensor | None = None
    num_samples_i = embedding_i.shape[0]
    num_samples_j = embedding_j.shape[0]
    def batch_generator() -> list[tuple[int, int, int, int]]:
        batch = []
        for i in range(1 + num_samples_i // batch_size):
            for j in range(1 + num_samples_j // batch_size):
                i0 : int = i * batch_size
                i1 : int = min(i0 + batch_size, num_samples_i)
                j0 : int = j *batch_size
                j1 : int = min(j0 + batch_size, num_samples_j)
                batch.append((i0, i1, j0, j1))
        return batch
    for i0, i1, j0, j1 in tqdm(batch_generator()):
        with torch.no_grad():
            num_i = i1 - i0
            num_j = j1 - j0
            d_i = embedding_i[i0 : i1]
            d_j = embedding_j[j0 : j1]
            distances : torch.Tensor = embedding.distance(d_i, d_j)
            flat_distances = distances.flatten()
            nearest_samples = torch.topk(-flat_distances, num_nearest)
            nearest_distances = - nearest_samples.values
            nearest_flat_index = nearest_samples.indices
            flat_index_i = einops.repeat(torch.arange(i0, i1), 'i -> i j', j=num_j).flatten()
            flat_index_j = einops.repeat(torch.arange(j0, j1), 'j -> i j', i=num_i).flatten()
            index_i = flat_index_i[nearest_flat_index]
            index_j = flat_index_j[nearest_flat_index]
            if min_distances is None:
                min_distances = nearest_distances
                min_index_i = index_i
                min_index_j = index_j
            else:
                min_distances = torch.cat((min_distances, nearest_distances))
                min_index_i = torch.cat((min_index_i, index_i))
                min_index_j = torch.cat((min_index_j, index_j))
    if no_identity_twice:
        num_remaining = min_distances.shape[0]
        identities_i=collection_data_i.identities[min_index_i]
        identities_j=collection_data_j.identities[min_index_j]
        mask = torch.ones((num_remaining,), device=embedding.device, dtype=torch.bool)
        for k0 in range(num_remaining):
            if not mask[k0]:
                continue
            id_i_0 = identities_i[k0]
            id_j_0 = identities_j[k0]
            for k1 in range(k0 + 1, num_remaining):
                if not mask[k1]:
                    continue
                id_i_1 = identities_i[k1]
                id_j_1 = identities_j[k1]
                if id_i_0 == id_i_1 or id_j_0 == id_j_1:
                    d_0 = min_distances[k0]
                    d_1 = min_distances[k1]
                    if d_0 < d_1:
                        mask[k1] = False
                    else:
                        mask[k0] = False
        to_keep = torch.arange(num_remaining, device=embedding.device)
        to_keep = torch.masked_select(to_keep, mask)
        min_distances = min_distances[to_keep]
        min_index_i = min_index_i[to_keep]
        min_index_j = min_index_j[to_keep]
    nearest_samples = torch.topk(-min_distances, num_nearest)
    min_distances = - nearest_samples.values
    nearest_flat_index = nearest_samples.indices
    index_i = min_index_i[nearest_flat_index]
    index_j = min_index_j[nearest_flat_index]
    click.echo(min_distances)
    return SamplePairs(
        identities_i=collection_data_i.identities[index_i],
        identities_j=collection_data_j.identities[index_j],
        distances=min_distances,
        labels_i=None,
        labels_j=None)
    
@click.command(help='Extract leakage from multiple synthtic databases')
@click.option('--add-database-directory', required=True, multiple=True)
@click.option('--image-directory-postfix', default='images_arcface_112x112')
@click.option('--postprocessor-config', type=click.Choice(Cropper.get_output_configs()), default='arcface')
@click.option('--output-directory', required=True)
@click.option('--reference-collection-path', required=True)
@click.option('--batch-size', default=4096, type=int)
@click.option('--num-nearest', default=10, type=int)
def main(
        add_database_directory : list[str],
        image_directory_postfix : str,
        postprocessor_config : str,
        output_directory : str,
        reference_collection_path : str,
        batch_size : int,
        num_nearest : int
        ) -> None:
    device = torch.device('cpu')
    dtype = torch.float32
    databases_directories = add_database_directory
    postprocessor = Cropper(
        input_config=Cropper.Config.dlib,
        output_config=postprocessor_config,
        device=device,
        dtype=dtype)
    plot_engine = Plot(device=device)
    embedding = Embedding(device=device)
    click.echo('Loading reference sample collection...')
    reference_collection_data = plot_engine.load_sample_collection(
        collection_name='ffhq',
        file_path=reference_collection_path)
    ffhq_database = FFHQDatabase()
    ffhq_samples : list[FFHQDatabase.Sample] = ffhq_database.query()
    click.echo('...done')
    for database_directory in databases_directories:
        click.echo(f'-> {database_directory}')
        if not os.path.exists(database_directory):
            click.echo('...directory does not exist')
            continue
        if not os.path.isdir(database_directory):
            click.echo('...not a directory')
            continue
        database_images_directory = os.path.join(database_directory, image_directory_postfix)
        if not os.path.exists(database_images_directory):
            click.echo('...images directory does not exist')
            continue
        if not os.path.isdir(database_images_directory):
            click.echo('...not a directory')
            continue
        sample_collection_path = os.path.join(database_directory, 'samples.h5')
        if not os.path.exists(sample_collection_path):
            click.echo('...cannot find sample collection')
            continue
        click.echo('loading sample collection...')
        dataset_name = os.path.basename(os.path.normpath(database_directory))
        click.echo(f'-> {dataset_name}')
        sample_collection_data = plot_engine.load_sample_collection(
            collection_name=dataset_name,
            file_path=sample_collection_path)
        sample_pairs = find_nearest_embeddings(
            collection_data_i=reference_collection_data,
            collection_data_j=sample_collection_data,
            embedding=embedding,
            num_nearest=num_nearest)
        images = []
        click.echo('loading reference images ...')
        ffhq_path = []
        for i in tqdm(range(num_nearest)):
            identity = sample_pairs.identities_i[i]
            for sample in ffhq_samples:
                if sample.identity == identity:
                    click.echo(sample.path)
                    ffhq_path.append(sample.path)
                    image = ffhq_database.load_sample(sample)
                    image = postprocessor.crop(image)
                    images.append(image)
        click.echo('loading synthtic images ...')
        syn_path = []
        for j in tqdm(range(num_nearest)):
            identity = sample_pairs.identities_j[j]
            image_path = os.path.join(database_images_directory, f'{identity:05}', 'reference.png')
            click.echo(image_path)
            syn_path.append(os.path.join(dataset_name, image_directory_postfix, f'{identity:05}', 'reference.png'))
            image = utils.load_image(file_path=image_path)
            images.append(image)
        images = torch.cat(tuple(images))
        file_path = os.path.join(output_directory, dataset_name + '.png')
        utils.save_image(image=images, file_path=file_path, num_rows=num_nearest)
        csv_path = os.path.join(output_directory, dataset_name + '.csv')
        with open(csv_path, 'w') as csv_file:
            csv_file.write('#score, ffhq_path, syn_db_path\n')
            for j in tqdm(range(num_nearest)):
                csv_file.write(f'{sample_pairs.distances[j]}, {ffhq_path[j]}, {syn_path[j]}\n')

if __name__ == '__main__':
    main()
