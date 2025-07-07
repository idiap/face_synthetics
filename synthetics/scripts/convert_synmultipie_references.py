#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was written by a human

import os
import h5py
from tqdm import tqdm
import torch

from synthetics import SampleCollection, Sample

base_path =  ...
datasets = [...]
original_file = 'metadata/identities/representations.h5'
network_type = 'stylegan2'
device = torch.device('cpu')

for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)
    original_file_path = os.path.join(dataset_path, original_file)
    sample_collection_path = os.path.join(dataset_path, 'samples.h5')
    sample_collection = SampleCollection(file_path=sample_collection_path, read_only=False)
    print('opening', original_file_path)
    print('------->', sample_collection_path)
    with h5py.File(original_file_path) as h5_data:
        identities = list(h5_data['w_latent'].keys())
        for identity in tqdm(identities):
            identity_key = identity + '_reference'
            w_latent = torch.tensor(h5_data['w_latent'][identity][identity_key])
            sample = Sample()
            sample.network_type = network_type
            sample.w_latent = w_latent
            sample_collection.add_sample(
                identity=int(identity),
                label='reference',
                sample=sample)
        sample_collection.save()

from synthetics import Cropper
cropper = Cropper(
    input_config=Cropper.Config.dlib,
    output_config=Cropper.Config.arcface)

cropper.crop()