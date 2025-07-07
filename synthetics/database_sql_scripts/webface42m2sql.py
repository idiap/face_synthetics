#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os
from tqdm import tqdm
from synthetics import Database

db_path = '.../resource/database/WebFace42M/'
group = 'world'
output = '.../user/dgeissbuhler/databases/webface42m.sqlite'

identities = []
num_files = 0

database = Database(read_only=False, db_file_path=output)
print(database.db_file_path)
protocol = database._add_protocol('webface42m')

for dir_name in os.listdir(db_path):
    if os.path.isdir(os.path.join(db_path, dir_name)):
        print(dir_name)
        group = database._add_group(protocol=protocol, group_name=dir_name)
        for sub_dir_name in tqdm(os.listdir(os.path.join(db_path, dir_name))):
            if os.path.isdir(os.path.join(db_path, dir_name, sub_dir_name)):
                id_str = sub_dir_name[4:]
                identity = int(id_str)
                assert identity not in identities 
                for file_name in os.listdir(os.path.join(db_path, dir_name, sub_dir_name)):
                    file_path = os.path.join(db_path, dir_name, sub_dir_name, file_name)
                    if os.path.isfile(file_path) and file_name[-4:] == '.jpg':
                        file_key = os.path.join(dir_name, sub_dir_name, file_name)[:-4]
                        key = os.path.join(dir_name, sub_dir_name, file_key)
                        key = key.replace('/', '_')
                        database._add_sample(
                            group=group,
                            key=key,
                            path=file_path,
                            identity=identity,
                            check_exist=False,
                            commit_session=False)
                        num_files += 1
        database.db_session.commit()
        print('num_files:', num_files)
        print('num_ids:', len(identities))
print('done')
