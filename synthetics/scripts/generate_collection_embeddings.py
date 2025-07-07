#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os

import click
import torch
import tqdm

from synthetics import SampleCollection, DatabaseGenerator

# ---

@click.command(help='Generate missing sample collection embeddings')
@click.option('--collection-directory', required=True)
def main(collection_directory : str) -> None:
    assert os.path.exists(collection_directory)
    collection_path = os.path.join(collection_directory, 'samples.h5')
    fixed_collection_path = os.path.join(collection_directory, 'samples_fixed.h5')
    assert os.path.exists(collection_path)
    click.echo(f'-> loading collection {collection_path}')
    device = torch.device('cuda')
    sample_collection = SampleCollection(
        file_path=collection_path,
        read_only=True)
    sample_collection.load(device=device)
    identities = sample_collection.list_identities()
    click.echo(f'-> found {len(identities)} identities')
    try:
        sample = sample_collection.get_sample(identity=0, label='reference')
        embedding_type = sample.embedding_type
        click.echo(f'-> embedding_type: {embedding_type}')
        assert embedding_type is not None, 'cannot infer embedding from reference'
        network_type = sample.network_type
        click.echo(f'-> network_type: {network_type}')
        assert network_type is not None, 'cannot infer network from reference'
        assert network_type.startswith('stylegan')
    except:
        raise RuntimeError('cannot load reference')
    click.echo('-> creating new collection...')
    fixed_sample_collection = SampleCollection(
        file_path=fixed_collection_path,
        read_only=False)
    click.echo('-> setting up generator...')
    database_generator = DatabaseGenerator(
        identities=[], # skip loading collection a second time
        network_type=network_type,
        embedding_type=embedding_type,
        root_directory=collection_directory,
        create_directories=False,
        generate_images=False,
        device=device)
    click.echo('-> generating embeddings...')
    for identity in tqdm.tqdm(identities):
        labels = sample_collection.list_identity_labels(identity=identity)
        for label in labels:
            sample = sample_collection.get_sample(identity=identity, label=label)
            if sample.embedding is None:
                w = sample.w_latent
                e = database_generator.embedding_from_w(w=w, detach=True)
                sample.embedding = e
                sample.embedding_type = embedding_type
            fixed_sample_collection.add_sample(
                identity=identity,
                label=label,
                sample=sample)
    click.echo('-> saving collection...')
    fixed_sample_collection.save()
    click.echo('... done')

# ---
if __name__ == '__main__':
    main()

# ---
    