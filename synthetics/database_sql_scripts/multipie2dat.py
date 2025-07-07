#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os
from tqdm import tqdm

# This script need an environement with the corresponding bob packages
# Then run it from the root of the synthetics repo after adapting hardcoded paths
from bob.bio.base.database import FileListBioDatabase
from bob.bio.face.database import MultipieDatabase

output_dir = '.../project/citerus/legal/databases_projections/multipie/'

print('protocol U:')
mp_db = MultipieDatabase(protocol='U')
group_names = mp_db.groups()
print(group_names)
samples = mp_db.all_samples(groups=group_names)
keys = []
with open(os.path.join(output_dir, 'original_protocol_U.dat'), 'w') as dat_file:
    for sample in tqdm(samples):
        key = sample.path.replace('/', '_')
        assert key not in keys
        keys.append(keys)
        dat_file.write(key + '\n')

mp_db = FileListBioDatabase(
    filelists_directory='.../home/dgeissbuhler/synthetics/synthetics/database_sql_scripts/project/protocols',
    name='multipie',
    protocol='E_lit',
    original_directory='.../resource/database/Multi-Pie/data/',
    original_extension='.png')
protocol_name='E_lit'
print('protocol', protocol_name)
group_names = mp_db.groups()
print(group_names)
samples = mp_db.all_files(groups=group_names)
keys = []
with open(os.path.join(output_dir, 'original_protocol_E_lit.dat'), 'w') as dat_file:
    for sample in tqdm(samples):
        key = sample.path.replace('/', '_')
        assert key not in keys
        keys.append(keys)
        dat_file.write(key + '\n')

mp_db = FileListBioDatabase(
    filelists_directory='.../home/dgeissbuhler/synthetics/synthetics/database_sql_scripts/project/protocols',
    name='multipie',
    protocol='P_center_lit',
    original_directory='.../resource/database/Multi-Pie/data/',
    original_extension='.png')
protocol_name='P_center_lit'
print('new protocol ->', protocol_name)
print('protocol', protocol_name)
group_names = mp_db.groups()
print(group_names)
samples = mp_db.all_files(groups=group_names)
keys = []
with open(os.path.join(output_dir, 'original_protocol_P_center_lit.dat'), 'w') as dat_file:
    for sample in tqdm(samples):
        key = sample.path.replace('/', '_')
        assert key not in keys
        keys.append(keys)
        dat_file.write(key + '\n')
        
print('done')
