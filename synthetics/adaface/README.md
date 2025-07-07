# Training Face Recognition from Scratch

The training script used in our experiments is based on [AdaFace](https://github.com/mk-minchul/AdaFace) repository.

## Installation
```sh
conda create --name face_rec scikit-image matplotlib pandas scikit-learn  python=3.11
conda activate face_rec
pip install -r requirements.txt
pip install torchvision
pip install mxnet-mkl==1.6.0 numpy==1.23.5
```

## Train 
### Preapring Dataset 
#### Preapring Validation Datasets
To prepare valiadtion datasets, you need to first download `agedb_30.bin`, `calfw.bin`, `cfp_ff.bin`, `cfp_fp.bin` , `cplfw.bin`, `lfw.bin` files. They are available in [InsightFace Dataset Zoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_). For example all of them are available along with CASIA-Webface files.

After downloading bin files, you need to make `memfiles` from bin files using `convert.py`:
```sh
python convert.py --rec_path <PATH_to_ROOT_OF_ALL_BIN_FILES> --make_image_files 
```
A complete script for preparing validation data is available in `scripts/prep_data_validation.sh`.


#### Preapring Train Datasets Using Image Folder Dataset
If you want to use your custom training dataset, prepare images in folder (as label) structure 
and change the `--data_root` and `--train_data_path` accordingly. The custom dataset should be located at `<data_root>/<train_data_path>`.  Note that `--train_data_path` should refer to a folder names `imgs`.

A sample script for preparing training data from DCFace is available in `prep_data_train_folder.sh`.

- Note: You need to enter the number of identities in `--custom_num_class`.

### Training Scripts
Sample training scripts are available in `scripts`. For example `scripts/prep_and_train.sh` contains a complete sample script for preparing datasets and training. To compound multiple datasets (such as references and variants) you can use `scripts/compound_multiple_datasets.py`. A sample script to train face recognition model is provided in `scripts/train.sh`.

- Note: Train and validation images should not been necessarily in a same dir as `data_root`.



## Validation

### High Quality Image Validation Sets (LFW, CFPFP, CPLFW, CALFW, AGEDB)
For evaluation on 5 HQ image validation sets with pretrained models,
refer to 
```
bash scripts/eval_5valsets.sh
```
### Mixed Quality Scenario (IJBB, IJBC Dataset)
For validation IJBB, IJBC , you can use the code in `./validation_mixed`. A sample script is available in `./scripts/eval_ijb.sh`.


## Inferece
- [IMPORTANT] Note that our implementation assumes that input to the model is `BGR` color channel as in `cv2` package. InsightFace models assume `RGB` color channel as in `PIL` package. So all our evaluation code uses `BGR` color channel with `cv2` package.

### Example using provided sample images
AdaFace takes input images that are preproccsed. 
The preprocessing step involves 
1. aligned with facial landmark (using MTCNN) and 
2. cropped to 112x112x3 size whose color channel is BGR order. 

We provide the code for performing the preprocessing step. 
For using pretrained AdaFace model for inference, 

1. Download the pretrained adaface model and place it in `pretrained/`

2. For using pretrained AdaFace on below 3 images, run 
```
python inference.py
```

|                              img1                              |                              img2                              |                                                           img3 |
|:--------------------------------------------------------------:|:--------------------------------------------------------------:|---------------------------------------------------------------:|
| <img src="face_alignment/test_images/img1.jpeg" width="215" /> | <img src="face_alignment/test_images/img2.jpeg" width="130" /> | <img src="face_alignment/test_images/img3.jpeg" width="191" /> |

The similarity score result should be 
```
tensor([[ 1.0000,  0.7334, -0.0655],
        [ 0.7334,  1.0000, -0.0277],
        [-0.0655, -0.0277,  1.0000]], grad_fn=<MmBackward0>)
```

### General Inference Guideline
In a nutshell, inference code looks as below.
```python
from face_alignment import align
from inference import load_pretrained_model, to_input

model = load_pretrained_model('ir_50')
path = 'path_to_the_image'
aligned_rgb_img = align.get_aligned_face(path)
bgr_input = to_input(aligned_rgb_img)
feature, _ = model(bgr_input)
```

- Note that AdaFace model is a vanilla pytorch model which takes in `bgr_input` which is 112x112x3 
torch tensor with BGR color channel whose value is normalized with `mean=0.5` and `std=0.5`, 
as in [to_input()](https://github.com/mk-minchul/AdaFace/blob/d8114b3ca8c54cd81ef59ac34c19eda1c548ca17/inference.py#L22)
- When preprocessing step produces error, it is likely that the MTCNN cannot find face in an image. 
Refer to [issues/28](https://github.com/mk-minchul/AdaFace/issues/28) for the discussion.