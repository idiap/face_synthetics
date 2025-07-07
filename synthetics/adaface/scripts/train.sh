# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

task_num='1' # task number 
n_ids=30000 # number of identities
train_data_d1='./sg2_n30k_ir50_r14_k10_w01_e001_it100_lang/images_arcface_112x112'
train_data_d2='./sg2_n30k_ir50_r14_disco_n64_rl14/images_arcface_112x112'

mkdir ./all_data
rm -rf ./all_data/train_$task_num
mkdir ./all_data/train_$task_num
python scripts/compound_multiple_datasets.py --final_path    ./all_data/train_$task_num/imgs \
                                             --list_datasets $train_data_d1   $train_data_d2  

python main.py \
    --data_root ./all_data \
    --train_data_path ./all_data/train_$task_num/ \
    --val_data_path ./validation/ \
    --prefix ir50_$task_num"_"adaface \
    --output_dir ./experiments \
    --custom_num_class $n_ids \
    --gpus 1 \
    --use_16bit \
    --arch ir_50 \
    --batch_size 256 \
    --num_workers 8 \
    --epochs 30 \
    --lr_milestones 12,20,26 \
    --lr 0.1 \
    --head adaface \
    --m 0.4 \
    --h 0.333 \
    --low_res_augmentation_prob 0.2 \
    --crop_augmentation_prob 0.2 \
    --photometric_augmentation_prob 0.2