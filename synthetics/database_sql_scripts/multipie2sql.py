#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

from tqdm import tqdm

# This script need an environement with the corresponding bob packages
# Then run it from the root of the synthetics repo after adapting hardcoded paths
from bob.bio.base.database import FileListBioDatabase
from bob.bio.face.database import MultipieDatabase

from synthetics.databases import Database as NewMultipieDatabase

output = '.../user/dgeissbuhler/databases/multipie.sqlite'
database = NewMultipieDatabase(read_only=False, db_file_path=output)

protocols = MultipieDatabase.protocols()

meta_type_camera = database._add_meta_data_type('camera', NewMultipieDatabase.MetaDataType.MetaDataTypeEnum.str)
meta_type_expression = database._add_meta_data_type('expression', NewMultipieDatabase.MetaDataType.MetaDataTypeEnum.str)
meta_type_img_type = database._add_meta_data_type('img_type', NewMultipieDatabase.MetaDataType.MetaDataTypeEnum.str)

for protocol_name in protocols:
    print('new protocol ->', protocol_name)
    protocol = database._add_protocol(protocol_name=protocol_name)
    mp_db = MultipieDatabase(protocol=protocol_name)
    group_names = mp_db.groups()
    for group_name in group_names:
        print('new group ->', group_name)
        group = database._add_group(protocol=protocol, group_name=group_name)
        samples = mp_db.all_samples(groups=[group_name])
        for sample in tqdm(samples):
            key = sample.path.replace('/', '_')
            path = sample.path + '.png'
            db_sample = database._add_sample(
                group=group,
                key=key,
                path=path,
                identity=int(sample.subject_id))
            database._add_meta_data(
                sample=db_sample,
                type=meta_type_camera,
                value=sample.camera,
                check_exist=True,
                commit_session=False)
            database._add_meta_data(
                sample=db_sample,
                type=meta_type_expression,
                value=sample.expression,
                check_exist=True,
                commit_session=False)
            database._add_meta_data(
                sample=db_sample,
                type=meta_type_img_type,
                value=sample.img_type,
                check_exist=True,
                commit_session=False)

mp_db = FileListBioDatabase(
    filelists_directory='.../home/dgeissbuhler/synthetics/synthetics/database_sql_scripts/project/protocols',
    name='multipie',
    protocol='E_lit',
    original_directory='.../resource/database/Multi-Pie/data/',
    original_extension='.png')
protocol_name='E_lit'
print('new protocol ->', protocol_name)
protocol = database._add_protocol(protocol_name=protocol_name)
group_names = mp_db.groups()
for group_name in group_names:
    if group_name == 'world':
        group_name == 'train'
    print('new group ->', group_name)
    group = database._add_group(protocol=protocol, group_name=group_name)
    samples = mp_db.all_files(groups=[group_name])
    for sample in tqdm(samples):
        key = sample.path.replace('/', '_')
        path = sample.path + '.png'
        database._add_sample(
            group=group,
            key=key,
            path=path,
            identity=int(sample.client_id))

mp_db = FileListBioDatabase(
    filelists_directory='.../home/dgeissbuhler/synthetics/synthetics/database_sql_scripts/project/protocols',
    name='multipie',
    protocol='P_center_lit',
    original_directory='.../resource/database/Multi-Pie/data/',
    original_extension='.png')
protocol_name='P_center_lit'
print('new protocol ->', protocol_name)
protocol = database._add_protocol(protocol_name=protocol_name)
group_names = mp_db.groups()
for group_name in group_names:
    if group_name == 'world':
        group_name == 'train'
    print('new group ->', group_name)
    group = database._add_group(protocol=protocol, group_name=group_name)
    samples = mp_db.all_files(groups=[group_name])
    for sample in tqdm(samples):
        key = sample.path.replace('/', '_')
        path = sample.path + '.png'
        database._add_sample(
            group=group,
            key=key,
            path=path,
            identity=int(sample.client_id))
        
print('done')
