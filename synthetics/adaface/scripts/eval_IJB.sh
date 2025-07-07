# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

data_root='./database/IJB_Suite/'
checkpoint='./pretrained/adaface_ir50.ckpt' 
cd validation_mixed
python validate_IJB_BC.py --dataset_name IJBB --data_root $data_root --arch ir_50 --ckpt_path $checkpoint --output_dir ./experiments_eval
python validate_IJB_BC.py --dataset_name IJBC --data_root $data_root --arch ir_50 --ckpt_path $checkpoint --output_dir ./experiments_eval