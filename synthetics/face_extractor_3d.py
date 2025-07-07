#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import os
import sys
import click
from dataclasses import dataclass

import torch

from .cropper import Cropper
from . import utils
from .utils import Sample

sys.path.insert(0, os.path.join(utils.source_path, 'synthetics/deep_3d_face_recon'))
from .deep_3d_face_recon.models import create_model
from .deep_3d_face_recon.models.bfm import ParametricFaceModel

# ---

class FaceExtractor3D:

    @dataclass
    class Options():
        add_image : bool = True
        # TODO load model from config
        bfm_folder : str = utils.config['bfm_folder']
        bfm_model : str = 'BFM_model_front.mat'
        camera_d : float = 10.0
        center : float = 112.0
        checkpoints_dir : str = './checkpoints'
        dataset_mode = None
        ddp_port : int = 12355
        display_per_batch : bool = True
        epoch : str = '20'
        eval_batch_nums : str = 'inf'
        focal : float = 1015.0
        gpu_ids : int = 0
        img_folder : str = 'examples'
        init_path : str = os.path.join(
            utils.source_path, 'checkpoints/init_model/resnet50-0676ba61.pth')
        isTrain : bool = False
        model : str = 'facerecon'
        name : str = 'face_recon'
        net_recon : str = 'resnet50'
        phase : str = 'test'
        suffix : str = ''
        use_ddp : bool = False
        use_last_fc : bool = False
        verbose : bool = False
        vis_batch_nums : int = 1
        world_size : int = 1
        z_far : float = 15.0
        z_near : float = 5.0

    # ---

    def __init__(
            self,
            device=torch.device('cuda'),
            dtype=torch.float32) -> None:
        self.device = device
        torch.cuda.set_device(device)
        self.dtype = dtype
        self.options = self.Options()
        self.model = create_model(self.options)
        self.__load_networks()
        self.model.device = device
        self.model.parallelize()
        self.model.eval()
        self.__rendered = False
        self.face_model = ParametricFaceModel(bfm_folder=self.options.bfm_folder)
        self.face_model.to(self.device)

    # ---
    # adapted from deep_3d_face_recon/models/base_model.py:BaseModel::load_networks
    def __load_networks(self) -> None:
        model_path = utils.get_model_path('deep3dfr')
        print('loading the model from %s' % model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        for name in self.model.model_names:
            if isinstance(name, str):
                net = getattr(self.model, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                net.load_state_dict(state_dict[name])

    # ---

    def extract(
            self,
            image : torch.Tensor,
            landmarks : torch.Tensor = None) -> dict:
        """ Extracts face model coefficients form image. """
        self.__rendered = False
        image = image/2. + .5
        if landmarks is None:
            data = {'imgs': image}
        else:
            data = {'imgs': image,
                    'lms': landmarks}
        self.model.set_input(data)
        # rewrite the forward pass
        with torch.no_grad():
            output_coeff = self.model.net_recon(self.model.input_img)
            self.model.facemodel.to(self.model.device)
            self.model.pred_vertex, self.model.pred_tex, self.model.pred_color, self.model.pred_lm \
                = self.model.facemodel.compute_for_render(output_coeff)
            self.model.pred_coeffs_dict = self.model.facemodel.split_coeff(output_coeff)
        return self.model.pred_coeffs_dict

    # ---

    def __render(self) -> None:
        with torch.no_grad():
            self.model.pred_mask, \
            self.model.pred_depth, \
            self.model.pred_face \
                = self.model.renderer(
                                self.model.pred_vertex,
                                self.model.facemodel.face_buf,
                                feat=self.model.pred_color)
        self.__rendered = True

    # ---

    def get_face_image(self) -> torch.Tensor:
        """ Returns a rendered face model, ``extract`` must be called before. """
        if not self.__rendered:
            self.__render()
        face = self.model.pred_face
        face = 2.0 * face - 1.0
        return face

    # ---

    def get_depth_map(self) -> torch.Tensor:
        """ Returns a rendered depth map, ``extract`` must be called before. """
        if not self.__rendered:
            self.__render()
        return self.model.pred_depth

    # ---

    def get_mask(self) -> torch.Tensor:
        """ Returns a pixel face mask, ``extract`` must be called before. """
        if not self.__rendered:
            self.__render()
        return self.model.pred_mask

    # ---

    def get_mesh(self) -> torch.Tensor:
        return None

    # ---
    # adapted from eg3d/preprocessing/3dface2idr.py
    def compute_pose(self) -> dict:
        """
            Compute pose, camera intrinsics and camera label,
            ``extract`` must be called before.
        """
        assert self.model.pred_coeffs_dict is not None

        angle = self.model.pred_coeffs_dict['angle']
        trans = self.model.pred_coeffs_dict['trans'][0]
        R = self.face_model.compute_rotation(angle)[0]
        trans[2] += -10
        c = -torch.tensordot(R, trans, dims=1)
        pose = torch.eye(4, device=self.device)
        pose[:3, :3] = R

        c *= 0.27 # normalize camera radius
        c[1] += 0.006 # additional offset used in submission
        c[2] += 0.161 # additional offset used in submission
        pose[0,3] = c[0]
        pose[1,3] = c[1]
        pose[2,3] = c[2]

        focal = 2985.29 # = 1015*1024/224*(300/466.285)#
        w = 1024#224
        h = 1024#224

        count = 0
        K = torch.eye(3, device=self.device)
        K[0][0] = focal
        K[1][1] = focal
        K[0][2] = w/2.0
        K[1][2] = h/2.0

        Rot = torch.eye(3, device=self.device)
        Rot[0, 0] = 1
        Rot[1, 1] = -1
        Rot[2, 2] = -1
        pose[:3, :3] = torch.matmul(pose[:3, :3], Rot)

        out = {}
        out['intrinsics'] = K
        out['pose'] = pose
        out['label'] = self.__get_label(pose=pose, intrinsics=K)
        return out

    # ---
    # adapted from eg3d/preprocessing/preprocess_cameras.py:fix_pose_orig
    def __fix_pose_orig(
            self,
            pose : torch.Tensor) -> torch.Tensor:
        pose = torch.clone(pose)
        location = pose[:3, 3]
        radius = torch.linalg.norm(location)
        pose[:3, 3] = pose[:3, 3]/radius * 2.7
        return pose

    # ---
    # adapted from eg3d/preprocessing/preprocess_cameras.py:fix_intrinsics
    def __fix_intrinsics(
            self,
            intrinsics : torch.Tensor
            ) -> torch.Tensor:
        intrinsics = torch.clone(intrinsics)
        assert intrinsics.shape == (3, 3), intrinsics
        intrinsics[0,0] = 2985.29/700
        intrinsics[1,1] = 2985.29/700
        intrinsics[0,2] = 1/2
        intrinsics[1,2] = 1/2
        assert intrinsics[0,1] == 0
        assert intrinsics[2,2] == 1
        assert intrinsics[1,0] == 0
        assert intrinsics[2,0] == 0
        assert intrinsics[2,1] == 0
        return intrinsics

    # ---
    # adapted from eg3d/preprocessing/preprocess_cameras.py
    def __get_label(
            self,
            pose : torch.Tensor,
            intrinsics : torch.Tensor) -> torch.Tensor:
        pose = self.__fix_pose_orig(pose)
        intrinsics = self.__fix_intrinsics(intrinsics)
        label = torch.cat([pose.reshape(-1), intrinsics.reshape(-1)])
        return label.unsqueeze(0)

# ---

@click.command(
    help="Extract a 3D face from an image")
@click.option(
    '--input',
    '-i',
    type=click.Path(exists=True, dir_okay=False),
    help='Input file',
    required=True)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(exists=True, file_okay=False),
    help='Output directory',
    required=True)
def extract_3d_face(
        input : str,
        output_dir : str):
    device = torch.device('cuda:0')
    basename_without_ext = os.path.splitext(os.path.basename(input))[0]
    image = utils.load_image(input, device=device)
    cropper = Cropper(
        input_config=Cropper.Config.dlib,
        output_config=Cropper.Config.deep3dfr,
        device=device)
    image = cropper.crop(image)
    file_path = os.path.join(output_dir, basename_without_ext + '_crop.png')
    utils.save_image(image, file_path)
    extractor = FaceExtractor3D(device=device)
    result = extractor.extract(image=image)
    file_path = os.path.join(output_dir, basename_without_ext + '.h5')
    pose = extractor.compute_pose()
    c_label = pose['label']
    sample = Sample(c_label=c_label, bfm_model=result)
    sample.save(file_path=file_path)
    face = extractor.get_face_image()
    file_path = os.path.join(output_dir, basename_without_ext + '_face.png')
    utils.save_image(face, file_path)
    mask = extractor.get_mask()
    file_path = os.path.join(output_dir, basename_without_ext + '_mask.png')
    utils.save_image(mask, file_path, normalize=True)
    mask_bool = mask.ge(0.5)
    depth_map = extractor.get_depth_map()
    depth_min = torch.masked_select(depth_map, mask_bool).amin()
    depth_max = torch.masked_select(depth_map, mask_bool).amax()
    depth_map = 2.0 * (depth_max - depth_map) / (depth_max - depth_min) - 1.0
    depth_map = torch.where(mask_bool, depth_map, 0.0)
    file_path = os.path.join(output_dir, basename_without_ext + '_depth.png')
    utils.save_image(depth_map, file_path)
    file_path = os.path.join(output_dir, basename_without_ext + '.obj')
    extractor.model.save_mesh(file_path)
