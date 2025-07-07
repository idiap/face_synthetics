# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l','--list_datasets', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('-p','--final_path', type=str, required=True)
args = parser.parse_args()

import glob
import os
from tqdm import tqdm

os.makedirs(args.final_path, exist_ok=True)

for d,dataset in enumerate(args.list_datasets):
    all_folders = glob.glob(f'{dataset}/*')
    print(f'dataset {d+1}: {dataset} \n{len(all_folders)} images')

    for f,folder in tqdm(enumerate(all_folders)):
        for i,img in enumerate(glob.glob(f'{folder}/*')):
            os.makedirs(f'{args.final_path}/{img.split(os.sep)[-2]}', exist_ok=True)
            os.system(f'ln -s {img} {args.final_path}/{img.split(os.sep)[-2]}/d{d+1}_{img.split(os.sep)[-1]}')
            # os.system(f'ln -s {img} {args.final_path}/{img.split(os.sep)[-2]}')

print(f"final dataset: {len(glob.glob(f'{args.final_path}/*'))}")