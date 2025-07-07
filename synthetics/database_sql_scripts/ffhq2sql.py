#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import json
from tqdm import tqdm
from synthetics import Database

json_file = ...
output = ...
f = open(json_file)
data = json.load(f)
database = Database(read_only=False, db_file_path=output)
print(database.db_file_path)
protocol = database._add_protocol('ffhq')
training = database._add_group(protocol=protocol, group_name='training')
validation = database._add_group(protocol=protocol, group_name='validation')

for id_str in tqdm(data.keys()):
    identity = int(id_str)
    group_name = data[id_str]['category']
    path_orig : str = data[id_str]['image']['file_path']
    # Idiap specific path
    path_items = path_orig.split('/')
    path = path_items[0] + '/' + path_items[1][0:2] + '/' + path_items[2]
    key = path.replace('/','_')[:-4]
    if group_name == 'training':
        group = training
    elif group_name == 'validation':
        group = validation
    database._add_sample(
        group=group,
        key=key,
        path=path,
        identity=identity,
        check_exist=False,
        commit_session=False)
    
database.db_session.commit()
print('done')
