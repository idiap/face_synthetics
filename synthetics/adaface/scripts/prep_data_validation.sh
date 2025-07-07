# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

mkdir all_data
mkdir all_data/validation

ln -s $train_data all_data/train/imgs
for val_dataset in {'agedb_30','calfw','cfp_ff','cfp_fp','cplfw','lfw'}
do
    ln -s ./CASIA-Webface/mxnet/$val_dataset.bin all_data/validation
done 

python convert.py --rec_path all_data/validation --make_validation_memfiles