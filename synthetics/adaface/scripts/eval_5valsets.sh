# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Hatef OTROSHI <hatef.otroshi@idiap.ch>
# SPDX-License-Identifier: MIT

cwd=$(pwd)

checkpoint='./pretrained/adaface_ir50.ckpt'

cd face_rec # face_rec is the repo dir
python main.py \
    --data_root $cwd/all_data \
    --train_data_path $cwd/all_data/train/ \
    --val_data_path $cwd/all_data/validation/ \
    --prefix adaface_ir50 \
    --gpus 1 \
    --use_16bit \
    --start_from_model_statedict $checkpoint \
    --arch ir_50 \
    --evaluate