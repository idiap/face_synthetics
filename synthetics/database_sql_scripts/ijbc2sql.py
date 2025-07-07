#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import csv
from tqdm import tqdm
from dataclasses import dataclass
from synthetics import Database

@dataclass
class ProtocolFile:
    protocol : Database.Protocol
    group : Database.Group
    path : str

output = ...

database = Database(read_only=False, db_file_path=output)
test1_verification = database._add_protocol('test1_verification')
test4_identification = database._add_protocol('test4_identification')
group_enroll = database._add_group(test1_verification, 'enroll')
group_verif = database._add_group(test1_verification, 'verif')
group_gallery_g1 = database._add_group(test4_identification, 'gallery_g1')
group_gallery_g2 = database._add_group(test4_identification, 'gallery_g2')
group_probes = database._add_group(test4_identification, 'probes')
metadata_type_template_id = database._add_meta_data_type('template_id', Database.MetaDataType.MetaDataTypeEnum.int)
metadata_type_face_x = database._add_meta_data_type('face_x', Database.MetaDataType.MetaDataTypeEnum.int)
metadata_type_face_y = database._add_meta_data_type('face_y', Database.MetaDataType.MetaDataTypeEnum.int)
metadata_type_face_w = database._add_meta_data_type('face_w', Database.MetaDataType.MetaDataTypeEnum.int)
metadata_type_face_h = database._add_meta_data_type('face_h', Database.MetaDataType.MetaDataTypeEnum.int)

protocol_files = \
[
    ProtocolFile(test1_verification, group_enroll, '.../resource/database/IJB-C/IJB/IJB-C/protocols/test1/enroll_templates.csv'),
    ProtocolFile(test1_verification, group_verif, '.../resource/database/IJB-C/IJB/IJB-C/protocols/test1/verif_templates.csv'),
    #ProtocolFile(test4_identification, group_gallery_g1, '.../resource/database/IJB-C/IJB/IJB-C/protocols/test4/gallery_G1.csv'),
    #ProtocolFile(test4_identification, group_gallery_g2, '.../resource/database/IJB-C/IJB/IJB-C/protocols/test4/gallery_G2.csv'),
    #ProtocolFile(test4_identification, group_probes, '.../resource/database/IJB-C/IJB/IJB-C/protocols/test4/probes.csv')
]

for protocol_file in protocol_files:
    print(protocol_file)
    num_samples = -1
    with open(protocol_file.path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(csv_reader):
            #TEMPLATE_ID,SUBJECT_ID,FILENAME,FACE_X,FACE_Y,FACE_WIDTH,FACE_HEIGHT
            num_samples += 1
            if num_samples == 0:
                continue
            template_id = int(row[0])
            subject_id = int(row[1])
            filename = row[2]
            face_x = int(row[3])
            face_y = int(row[4]) 
            face_w = int(row[5])
            face_h = int(row[6])
            file_key = int(filename.split('/')[1][:-4])
            key = f'{file_key:09d}_{template_id:09d}_{face_x:03d}_{face_y:03d}_{face_w:03d}_{face_h:03d}'
            sample = database._add_sample(
                group=protocol_file.group,
                key=key,
                path=filename,
                identity=subject_id,
                check_exist=False,
                commit_session=False)
            database._add_meta_data(
                sample=sample,
                type=metadata_type_template_id,
                value=template_id,
                check_exist=False,
                commit_session=False)
            database._add_meta_data(
                sample=sample,
                type=metadata_type_face_x,
                value=face_x,
                check_exist=False,
                commit_session=False)
            database._add_meta_data(
                sample=sample,
                type=metadata_type_face_y,
                value=face_y,
                check_exist=False,
                commit_session=False)
            database._add_meta_data(
                sample=sample,
                type=metadata_type_face_w,
                value=face_w,
                check_exist=False,
                commit_session=False)
            database._add_meta_data(
                sample=sample,
                type=metadata_type_face_y,
                value=face_y,
                check_exist=False,
                commit_session=False)
            if num_samples % 256 == 0:
                database.db_session.commit()
    database.db_session.commit()
print('done')
