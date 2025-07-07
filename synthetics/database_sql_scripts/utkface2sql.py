#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import json
from tqdm import tqdm
from synthetics import Database
import pandas
import os

lst_train = ".../bob.project.margin/bob/project/margin/lists/utk/ageasis/norm/train_world.lst"
lst_test = ".../bob.project.margin/bob/project/margin/lists/utk/ageasis/eval/for_probes.lst"
output = ".../face-synthetics/data_path/databases_index/utkface_ageasis_train_subset_balanced_fullset_test.sqlite"
data_root = '../margin/datasets/utkface-upscaled/'
genders = ['male', 'female']
races = ['White', 'Black', 'Asian', 'Indian', 'Others']

sampling_proportion = 0.09 #limited by number of children after filtering
filter_age = True

df_train = pandas.read_csv(lst_train,  sep=" ", header=None, names=['path', 'age'])
df_train["set"] = 'training'
df_test = pandas.read_csv(lst_test,  sep=" ", header=None, names=['path', 'age'])
df_test["set"] = 'testing'

if filter_age:
    df_train = df_train.query('(age > 7 & age < 18) | age > 25')
    #df_test = df_test.query('(age > 7 & age < 18) | age > 25')
    #print(df_train)
    #print(df_test)

if sampling_proportion != 1.0:
    print(len(df_train))
    print(df_train.query('age < 18'))
    print(sampling_proportion*len(df_train)*0.5)

    df_train_kids = df_train.query('age < 18').sample(int(0.5*sampling_proportion*len(df_train)))
    df_train_adults = df_train.query('age > 25').sample(int(0.5*sampling_proportion*len(df_train)))
    
    df_train = pandas.concat([df_train_kids, df_train_adults], ignore_index=True)

    print(len(df_test))
    #print(df_test.query('age < 18'))
    #print(sampling_proportion*len(df_test)*0.5)

    #df_test_kids = df_test.query('age < 18').sample(int(0.5*sampling_proportion*len(df_test)))
    #df_test_adults = df_test.query('age > 25').sample(int(0.5*sampling_proportion*len(df_test)))
    

    #df_test = pandas.concat([df_test_kids, df_test_adults], ignore_index=True)


df_all = pandas.concat([df_train, df_test], ignore_index=True)


database = Database(read_only=False, db_file_path=output)
print(database.db_file_path)

protocol = database._add_protocol('utkface_ageasis')
training = database._add_group(protocol=protocol, group_name='training')
testing = database._add_group(protocol=protocol, group_name='testing')

meta_type_age = database._add_meta_data_type('age', database.MetaDataType.MetaDataTypeEnum.int)
meta_type_gender = database._add_meta_data_type('gender', database.MetaDataType.MetaDataTypeEnum.str)
meta_type_race = database._add_meta_data_type('race', database.MetaDataType.MetaDataTypeEnum.str)

#meta_type_camera = database._add_meta_data_type('underage', database.MetaDataType.MetaDataTypeEnum.int) #underage < 18

for i in tqdm(range(len(df_all))):
    
    #get main attributes
    identity = (i)
    group_name = df_all.at[i, "set"]
    img_path = df_all.at[i,"path"]
    key = img_path.replace('/','_')
    #print(data_root)
    #print(df_all.at[i, "path"] + ".jpg")
    path = os.path.join(data_root, img_path + ".jpg")
    if not (os.path.exists(path)):
        print(f"path {path} does not exist, trying with extension in caps")
        path = path[:-4] + '.JPG'
        if not(os.path.exists(path)):
            print(f"path {path} does not exist, skipping")
            continue
    
    if group_name == 'training':
        group = training
    elif group_name == 'testing':
        group = testing

    #get meta attributes
    age = int(df_all.at[i, "age"])
    #print(img_path.split('/')[-1])
    #print(((img_path.split('/')[-1]).split('_'))[2])
    gender = genders[int(((img_path.split('/')[-1]).split('_'))[1])]
    race = races[int(((img_path.split('/')[-1]).split('_'))[2])]
    


    #commit to db
    db_sample = database._add_sample(
        group=group,
        key=key,
        path=path,
        identity=identity,
        check_exist=False,
        commit_session=False
        )
    database._add_meta_data(
                sample=db_sample,
                type=meta_type_age,
                value=age,
                check_exist=True,
                commit_session=False)
    database._add_meta_data(
                sample=db_sample,
                type=meta_type_gender,
                value=gender,
                check_exist=True,
                commit_session=False)
    database._add_meta_data(
                sample=db_sample,
                type=meta_type_race,
                value=race,
                check_exist=True,
                commit_session=False)

    
database.db_session.commit()
print('done')