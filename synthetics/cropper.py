#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

from dataclasses import dataclass
from enum import Enum, unique

import click

import torch
import torchvision
import einops
import kornia
from scipy.io import loadmat

from .landmark_detector import LandmarkDetector
from . import utils

# ---

@unique
class CropAlgorithm(Enum):
    NOCROP = 0
    RESIZE = 1
    FIXED_2_POINTS = 2
    FFHQ = 3
    EG3D = 4
    ARCFACE = 5

# ---

class Cropper():

    class Config(Enum):
        @dataclass
        class ConfigItem:
            algorithm : CropAlgorithm | None = None
            height : int | None = None
            width : int | None = None
            eye_left : tuple[float, float] | None = None
            eye_right : tuple[float, float] | None = None
            nose : tuple[float, float] | None = None
            mouth_left : tuple[float, float] | None = None
            mouth_right : tuple[float, float] | None = None
            rescale_factor : float | None = None
            center_crop_size : int | None = None
            input : bool = False
            output : bool = False
            landmark_detector_backend : LandmarkDetector.Backend | None = None
            _key : int | None = None # used in case enum not listed correctly
        ffhq = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=1024,
            width=1024,
            eye_left=(650., 480.),
            eye_right=(380., 480.),
            input=True,
            output=True)
        ffhq256 = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=256,
            width=256,
            eye_left=(162.5, 120.),
            eye_right=(95., 120.),
            input=True,
            output=True)
        ffhq128 = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=128,
            width=128,
            eye_left=(81.25, 60.),
            eye_right=(47.5, 60.),
            input=True,
            output=True)
        synmultipie = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=182,
            width=182,
            eye_left=(121.33, 60.67),
            eye_right=(60.67, 60.67),
            output=True,
            input=True)
        synface = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=256,
            width=256,
            eye_left=(162.5, 120.),
            eye_right=(95., 120.),
            output=True,
            input=True,
            _key = 1)
        sf2_feat = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=160,
            width=160,
            eye_left=(107., 46.),
            eye_right=(53., 46.),
            output=True,
            input=True)
        iresnet_bob = ConfigItem(
            algorithm=CropAlgorithm.FFHQ,
            height=112,
            width=112,
            eye_left=(73.5318, 51.6),
            eye_right=(38.2946, 51.6),
            output=True,
            input=True)
        arcface = ConfigItem(
            algorithm=CropAlgorithm.ARCFACE,
            height=112,
            width=112,
            eye_left=(73.53179932, 51.50139999),
            eye_right=(38.29459953, 51.69630051),
            nose=(56.02519989, 71.73660278),
            mouth_left=(70.72990036, 92.20410156),
            mouth_right=(41.54930115, 92.3655014),
            input=False,
            output=True)
        eg3d = ConfigItem(
            algorithm=CropAlgorithm.EG3D,
            height=512,
            width=512,
            rescale_factor=300.,
            center_crop_size=700,
            input=False,
            output=True)
        deep3dfr = ConfigItem(
            algorithm=CropAlgorithm.EG3D,
            height=224,
            width=224,
            rescale_factor=102.,
            center_crop_size=224,
            input=False,
            output=True)
        mtcnn = ConfigItem(
            algorithm=None,
            input=True,
            landmark_detector_backend=LandmarkDetector.Backend.MTCNN)
        dlib = ConfigItem(
            algorithm=None,
            input=True,
            landmark_detector_backend=LandmarkDetector.Backend.DLIB)
        kornia = ConfigItem(
            algorithm=None,
            input=True,
            landmark_detector_backend=LandmarkDetector.Backend.KORNIA)
        nocrop = ConfigItem(
            algorithm=CropAlgorithm.NOCROP,
            input=False,
            output=True)
        resize = ConfigItem(
            algorithm=CropAlgorithm.RESIZE,
            input=True,
            output=False)

    # ---

    @staticmethod
    def get_input_configs() -> list[str]:
        return [config.name for config in Cropper.Config if config.value.input]

    @staticmethod
    def get_output_configs() -> list[str]:
        return [config.name for config in Cropper.Config if config.value.output]

    @staticmethod
    def get_config_item(
            config : str | Config
            ) -> Config.ConfigItem:
        if isinstance(config, Cropper.Config):
            return config.value
        elif isinstance(config, str):
            for config_item in Cropper.Config:
                if config == config_item.name:
                    return config_item.value
            raise RuntimeError('Unknown cropper config')
        else:
            raise RuntimeError('Incorrect type')

    # ---

    def __init__(
            self,
            input_config : str | Config,
            output_config : str | Config,
            device = torch.device('cuda'),
            dtype = torch.float32,
            ) -> None:
        self.input_config = self.get_config_item(input_config)
        assert self.input_config.input
        self.output_config = self.get_config_item(output_config)
        assert self.output_config.output
        self.device = device
        self.dtype = dtype
        if self.input_config.algorithm is None:
            self.algorithm = self.output_config.algorithm
        elif self.input_config.algorithm == CropAlgorithm.RESIZE:
            self.algorithm = CropAlgorithm.RESIZE
        else:
            self.algorithm = CropAlgorithm.FIXED_2_POINTS
        if self.algorithm == CropAlgorithm.EG3D:
            self.standard_3d_landmarks = self.__load_lm3d()
        else:
            self.standard_3d_landmarks = None
        if self.output_config.algorithm == CropAlgorithm.NOCROP:
            self.algorithm = CropAlgorithm.NOCROP
        self.input_width = self.input_config.width
        self.input_height = self.input_config.height
        self.output_width = self.output_config.width
        self.output_height = self.output_config.height
        if self.input_config.landmark_detector_backend is not None:
            self.landmarks_detector = LandmarkDetector(
                backend=self.input_config.landmark_detector_backend,
                device=self.device,
                dtype=self.dtype)
        else:
            self.landmarks_detector = None
        if self.algorithm == CropAlgorithm.FIXED_2_POINTS:
            self.input_width = self.input_config.width
            self.input_height = self.input_config.height
            input_eye_left = torch.tensor(
                self.input_config.eye_left,
                device=self.device)
            input_eye_right = torch.tensor(
                self.input_config.eye_right,
                device=self.device)
            assert input_eye_left[0] > input_eye_right[0], \
                'Eyes position swapped'
            output_eye_left = torch.tensor(
                self.output_config.eye_left,
                device=self.device)
            output_eye_right = torch.tensor(
                self.output_config.eye_right,
                device=self.device)
            assert output_eye_left[0] > output_eye_right[0], \
                'Eyes position swapped'
            self.scaling_factor = \
                (output_eye_left[0] - output_eye_right[0]) \
                / (input_eye_left[0]- input_eye_right[0])
            coord_upper_left_corner = input_eye_right \
                - 1 / self.scaling_factor * output_eye_right
            coord_lower_right_corner = coord_upper_left_corner \
                + torch.tensor(
                    (self.output_width, self.output_height),
                    device=self.device) \
                / self.scaling_factor
            self.upper_left_corner_x = int(coord_upper_left_corner[0])
            self.upper_left_corner_y = int(coord_upper_left_corner[1])
            self.lower_right_corner_x = int(coord_lower_right_corner[0])
            self.lower_right_corner_y = int(coord_lower_right_corner[1])
            assert self.upper_left_corner_y >= 0
            assert self.upper_left_corner_x >= 0
            if self.input_height is not None:
                assert self.lower_right_corner_y <= self.input_height
            if self.input_width is not None:
                assert self.lower_right_corner_x <= self.input_width
        else:
            self.input_width = None
            self.input_height = None
            self.scaling_factor = None
            self.upper_left_corner_y = None
            self.upper_left_corner_x = None
            self.lower_right_corner_y = None
            self.lower_right_corner_x = None

    # ---

    def __get_landmarks(
            self,
            image : torch.Tensor
            ) -> LandmarkDetector.FaceLandmarks | None:
        """ Get landmarks and fix errors. """
        if self.landmarks_detector is None:
            return None
        face_landmarks = self.landmarks_detector.detect(image)
        if face_landmarks is None:
            return None
        if face_landmarks is not None and any(face_landmarks.error):
            click.echo('WARNING: landmark detection error')
            # HACK replace faulty landmarks by default arcface ones
            # This avoids fatal errors and, in some case, recovery
            # TODO there should be a better way
            batch_size = image.shape[0]
            width = image.shape[3]
            height = image.shape[2]
            scale_x = float(width)/float(Cropper.Config.arcface.value.width)
            scale_y = float(height)/float(Cropper.Config.arcface.value.height)
            default_eye_left = torch.tensor(
                (Cropper.Config.arcface.value.eye_left[0] * scale_x,
                 Cropper.Config.arcface.value.eye_left[1] * scale_y),
                device=self.device)
            default_eye_right = torch.tensor(
                (Cropper.Config.arcface.value.eye_right[0] * scale_x,
                 Cropper.Config.arcface.value.eye_right[1] * scale_y),
                device=self.device)
            default_nose = torch.tensor(
                (Cropper.Config.arcface.value.nose[0] * scale_x,
                 Cropper.Config.arcface.value.nose[1] * scale_y),
                device=self.device)
            default_mouth_left = torch.tensor(
                (Cropper.Config.arcface.value.mouth_left[0] * scale_x,
                 Cropper.Config.arcface.value.mouth_left[1] * scale_y),
                device=self.device)
            default_mouth_right = torch.tensor(
                (Cropper.Config.arcface.value.mouth_right[0] * scale_x,
                 Cropper.Config.arcface.value.mouth_right[1] * scale_y),
                device=self.device)
            for b in range(batch_size):
                if face_landmarks.error[b]:
                    face_landmarks.eye_left[b, :] = default_eye_left
                    face_landmarks.eye_right[b, :] = default_eye_right
                    face_landmarks.nose[b, :] = default_nose
                    face_landmarks.mouth_left[b, :] = default_mouth_left
                    face_landmarks.mouth_right[b, :] = default_mouth_right
        return face_landmarks

    # ---

    def crop(
            self,
            image : torch.Tensor,
            face_landmarks : LandmarkDetector.FaceLandmarks | None = None,
            ) -> torch.Tensor:
        """
            Crop an image (3D torch Tensor) or an image batch (4D torch Tensor).
        """
        assert torch.is_tensor(image)
        assert image.ndim == 4 or image.ndim == 3
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeezed_batch_index = True
        else:
            squeezed_batch_index = False
        if face_landmarks is None and self.landmarks_detector is not None:
            face_landmarks = self.__get_landmarks(image=image)
        if self.algorithm == CropAlgorithm.NOCROP:
            image = image
        elif self.algorithm == CropAlgorithm.RESIZE:
            image = self.__resize(
                image=image,
                width=self.output_config.width,
                height=self.output_config.height)
        elif self.algorithm == CropAlgorithm.FIXED_2_POINTS:
            image = self.__crop_fixed(image)
        elif self.algorithm == CropAlgorithm.FFHQ:
            image = self.__crop_ffhq(image, face_landmarks)
        elif self.algorithm == CropAlgorithm.EG3D:
            image = self.__crop_eg3d(image, face_landmarks)
        elif self.algorithm == CropAlgorithm.ARCFACE:
            image = self.__crop_arcface(image, face_landmarks)
        else:
            raise RuntimeError('Unknown crop algorithm')
        if squeezed_batch_index:
            return image[0]
        else:
            return image

    # ---
    def __crop_fixed(
            self,
            image : torch.Tensor
            ) -> torch.Tensor:
        assert image.shape[2] == self.input_height
        assert image.shape[3] == self.input_width
        image = image[:, :,
            self.upper_left_corner_y : self.lower_right_corner_y,
            self.upper_left_corner_x : self.lower_right_corner_x]
        image = self.__resize(
            image=image,
            width=self.output_width,
            height=self.output_height)
        return image

    # ---
    # adapted from github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    def __crop_ffhq(
            self,
            image : torch.Tensor,
            face_landmarks : LandmarkDetector.FaceLandmarks,
            oversampling=4,
            enable_padding=True
            ) -> torch.Tensor:
        """
            Crop, resize and align the image based on the detected landmarks,
            in the same way FFHQ has been preprocessed for training StyleGAN2.
        """
        assert torch.is_tensor(image)
        assert image.ndim == 4
        assert image.shape[1] == 3
        if any(face_landmarks.error):
            raise RuntimeError("Landmark detection error")
        num_images = image.shape[0]
        img_size_y = image.shape[2]
        img_size_x = image.shape[3]
        if face_landmarks.landmarks_68p is not None:
            # Keep old eye convention for DLIB
            lm_eye_left = face_landmarks.landmarks_68p[:, 42 : 48]
            lm_eye_right = face_landmarks.landmarks_68p[:, 36 : 42]
            lm_mouth_outer = face_landmarks.landmarks_68p[:, 48 : 60]
            eye_left = torch.mean(lm_eye_left, axis=1)
            eye_right = torch.mean(lm_eye_right, axis=1)
            mouth_left = lm_mouth_outer[:, 6]
            mouth_right = lm_mouth_outer[:, 0]
        else:
            # MTCNN and KORNIA (might lead to different results)
            eye_left = face_landmarks.eye_left
            eye_right = face_landmarks.eye_right
            mouth_left = face_landmarks.mouth_left
            mouth_right = face_landmarks.mouth_right
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_left - eye_right
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle
        def rotate(v):
            return torch.cat((-v[:, 1].unsqueeze(1), v[:, 0].unsqueeze(1)), 1)
        x = eye_to_eye - rotate(eye_to_mouth)
        h = torch.hypot(x[:,0], x[:,1])
        h = einops.repeat(h, 'b -> b d', d=2)
        x = torch.div(x, h)
        fact_eye_to_eye = torch.hypot(eye_to_eye[:,0], eye_to_eye[:,1]) * 2.0
        fact_eye_to_mouth = torch.hypot(eye_to_mouth[:,0], eye_to_mouth[:,1]) * 1.8
        m = torch.maximum(fact_eye_to_eye, fact_eye_to_mouth)
        m = einops.repeat(m, 'b -> b d', d=2)
        x = torch.mul(x, m)
        y = rotate(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = torch.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        quad = einops.rearrange(quad, 'n b d -> b n d')
        qsize = torch.hypot(x[:,0], x[:,1]) * 2

        # Shrink
        # TODO: vectorize and activate
        # shrink = torch.floor(qsize / self.output_size * 0.5).int()
        # if shrink > 1:
        #     rsize_x = int(torch.round(float(img_size_x) / shrink))
        #     rsize_y = int(torch.round(float(img_size_y) / shrink))
        #     resize  = torchvision.transforms.Resize((rsize_y, rsize_x), antialias=True)
        #     img = resize(img)
        #     print('shrink inner', img.shape)

        #     quad /= shrink
        #     qsize /= shrink

        # Crop
        border = torch.clamp(torch.round(qsize * 0.1).int(), min=3)
        quad_min_x = torch.floor(torch.amin(quad[:, :, 0], 1)).int()
        quad_max_x = torch.ceil(torch.amax(quad[:, :, 0], 1)).int()
        quad_min_y = torch.floor(torch.amin(quad[:, :, 1], 1)).int()
        quad_max_y = torch.ceil(torch.amax(quad[:, :, 1], 1)).int()

        left = torch.clamp(quad_min_x - border, min=0)
        top = torch.clamp(quad_min_y - border, min=0)
        right = torch.clamp(quad_max_x + border, max=img_size_x)
        bottom = torch.clamp(quad_max_y + border, max=img_size_y)
        width = right - left
        height = bottom - top

        # TODO vectorize
        images = []
        shifts  = []
        for i in range(num_images):
            img = image[i]
            if width[i] < img_size_x or height[i] < img_size_y:
                img = torchvision.transforms.functional.crop(
                    img,
                    top=top[i],
                    left=left[i],
                    height=height[i],
                    width=width[i])
                shift = torch.tensor(
                    [left[i], top[i]],
                    device=self.device,
                    dtype=self.dtype)
                shift = einops.repeat(shift, 'd -> n d', n=4)
                shifts.append(shift)
            else:
                shift = torch.tensor(
                    [0, 0],
                    device=self.device,
                    dtype=self.dtype)
                shift = einops.repeat(shift, 'd -> n d', n=4)
                shifts.append(shift)
            images.append(img)

        shifts = torch.stack(shifts)
        quad -= shifts

        # Pad
        # pad_min_x = torch.floor(torch.amin(quad[:, :, 0], 1)).int()
        # pad_max_x = torch.ceil(torch.amax(quad[:, :, 0], 1)).int()
        # pad_min_y = torch.floor(torch.amin(quad[:, :, 1], 1)).int()
        # pad_max_y = torch.ceil(torch.amax(quad[:, :, 1], 1)).int()

        # pad_left    = torch.clamp(-pad_min_x + border, min=0)
        # pad_top     = torch.clamp(-pad_min_y + border, min=0)
        # pad_right   = torch.clamp(pad_max_x - img_size_x + border, min=0)
        # pad_bottom  = torch.clamp(pad_max_y - img_size_x + border, min=0)

        # TODO pad
        #for i in range(num_images):
            # if enable_padding and max(pad) > border - 4:

            #     bound = int(torch.round(qsize * 0.3))
            #     print('bound', bound)
            #     pad= (  max(pad[1], bound),
            #             max(pad[3], bound),
            #             max(pad[0], bound),
            #             max(pad[2], bound))
            #     print('pad')
            #     print(pad)
            #     #((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            #     img = torch.nn.functional.pad(
            #             img,
            #             pad=pad,
            #             mode='reflect')
                # y, x = torch.meshgrid[:1, :img_size_y, :img_size_x]
                # mask = torch.maximum(
                #         1.0 - torch.minimum(
                #                 torch.float32(x) / pad[0],
                #                 torch.float32(img_size_x-1-x) / pad[2]),
                #         1.0 - torch.minimum(
                #                 torch.float32(y) / pad[1],
                #                 torch.float32(img_size_y-1-y) / pad[3]))
                # blur = qsize * 0.02
                # gaussian_filter = torchvision.transforms.GaussianBlur([0, blur, blur])
                # img += (gaussian_filter(img, [blur, blur, 0]) - img) * torch.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                # img += (torch.median(img, axis=(0,1)) - img) * torch.clip(mask, 0.0, 1.0)
                #img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
                # quad += pad[:2]

            # Transform.
            #img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
            # [[top-left], [top-right], [bottom-right], [bottom-left]]

        # TODO image quality better with legacy (transform_size??)
        # TODO try anti-aliasing
        output_size = self.output_width # TODO better
        transform_size = int(output_size * oversampling)
        for i in range(num_images):
            img = images[i]
            size_x = img.shape[2]
            size_y = img.shape[1]
            resize = torchvision.transforms.Resize(
                (transform_size, transform_size),
                antialias=True)
            img = resize(img)
            quad[i, :, 0] *= float(transform_size)/float(size_x)
            quad[i, :, 1] *= float(transform_size)/float(size_y)
            source  = torch.round(quad[i]).tolist()
            dest    = [ [0,                 0],
                        [0,                 transform_size],
                        [transform_size,    transform_size],
                        [transform_size,    0]]
            img = torchvision.transforms.functional.perspective(
                img,
                source,
                dest,
                torchvision.transforms.InterpolationMode.BILINEAR)
            resize  = torchvision.transforms.Resize(
                (self.output_height, self.output_width),
                antialias=True)
            img = resize(img)
            images[i] = img
        return torch.stack(images)

    # ---
    # adapted from deep_3dfr/util/load_mats.py:load_lm3d()
    def __load_lm3d(
            self
        ) -> torch.Tensor:
        """ Loads 3D standard landmarks (BFM). """
        Lm3D = loadmat(utils.get_model_path('lm3d_std'))
        Lm3D = Lm3D['lm']
        Lm3D = torch.tensor(Lm3D, device=self.device, dtype=self.dtype)
        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = torch.tensor([31, 37, 40, 43, 46, 49, 55]) - 1
        Lm3D = torch.stack(
            [Lm3D[lm_idx[0], :],
            torch.mean(Lm3D[lm_idx[[1, 2]], :], 0),
            torch.mean(Lm3D[lm_idx[[3, 4]], :], 0),
            Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]],
            axis=0)
        Lm3D = Lm3D[[1, 2, 0, 3, 4], :]
        return Lm3D

    # ---
    # adapted from deep_3d_face_recon/util/preprocess.py:POS()
    def __POS(self, xp, x):
        npts = xp.shape[1]
        A = torch.zeros([2*npts, 8], device=self.device, dtype=self.dtype)
        A[0:2*npts-1:2, 0:3] = x.t()
        A[0:2*npts-1:2, 3] = 1
        A[1:2*npts:2, 4:7] = x.t()
        A[1:2*npts:2, 7] = 1
        b = torch.reshape(xp.t(), [2*npts, 1])
        k, _, _, _ = torch.linalg.lstsq(A, b)
        R1 = k[0:3]
        R2 = k[4:7]
        sTx = k[3]
        sTy = k[7]
        s = (torch.linalg.norm(R1) + torch.linalg.norm(R2))/2
        t = torch.stack([sTx, sTy], axis=0)
        return t, s

    # ---
    # from EG3D/preprocessing/crop_images_in_the_wild.py
    # and from deep_3d_face_recon/util/preprocess.py:align_img()
    # and from deep_3d_face_recon/util/preprocess.py:resize_n_crop_img()
    def __crop_eg3d(
            self,
            image : torch.Tensor,
            face_landmarks : LandmarkDetector.FaceLandmarks
            ) -> torch.Tensor:
        height_0 = image.shape[2]
        width_0 = image.shape[3]
        if face_landmarks.landmarks_68p is not None:
            landmarks_68p = face_landmarks.landmarks_68p
            landmarks_68p[:, :, 1] = height_0 - 1 - landmarks_68p[:, :, 1]
            lm_idx = torch.tensor([31, 37, 40, 43, 46, 49, 55]) - 1
            landmarks_5p = torch.stack(
                [landmarks_68p[:, lm_idx[0], :],
                torch.mean(landmarks_68p[:, lm_idx[[1, 2]], :], 1),
                torch.mean(landmarks_68p[:, lm_idx[[3, 4]], :], 1),
                landmarks_68p[:, lm_idx[5], :],
                landmarks_68p[:, lm_idx[6], :]],
                axis=1)
            landmarks_5p = landmarks_5p[:, [1, 2, 0, 3, 4], :]
        else:
            landmarks_5p : torch.Tensor = torch.stack([
                face_landmarks.eye_right,
                face_landmarks.eye_left,
                face_landmarks.nose,
                face_landmarks.mouth_right,
                face_landmarks.mouth_left],
                axis=1)
            landmarks_5p[:, :, 1] = height_0 - 1 - landmarks_5p[:, :, 1]
            landmarks_5p.to(self.device)
        assert self.output_config.height == self.output_config.width
        target_size = self.output_config.height
        assert self.output_config.rescale_factor is not None
        rescale_factor = self.output_config.rescale_factor
        assert self.output_config.center_crop_size is not None
        center_crop_size = self.output_config.center_crop_size

        # TODO batch mode
        assert landmarks_5p.ndim == 3 and landmarks_5p.shape[0] == 1
        assert image.shape[0] == 1
        landmarks_5p = landmarks_5p[0]
        # calculate translation and scale factors using 5 facial landmarks and
        # standard landmarks of a 3D face
        t, s = self.__POS(landmarks_5p.t(), self.standard_3d_landmarks.t())
        s = rescale_factor/s
        width = (width_0 * s).to(torch.int32)
        height = (height_0 * s).to(torch.int32)
        left = (width/2 - target_size/2 + float((t[0] - width_0/2)*s)).to(torch.int32)
        up = (height/2 - target_size/2 + float((height_0/2 - t[1])*s)).to(torch.int32)

        width=int(width)
        height=int(height)
        left=int(left) + int(target_size/2 - center_crop_size/2)
        up=int(up) + int(target_size/2 - center_crop_size/2)

        image = self.__resize(
            image,
            width=width,
            height=height)
        image = self.__crop(
            image=image,
            left=left,
            width=center_crop_size,
            up=up,
            height=center_crop_size)
        image = self.__resize(
            image,
            width=self.output_width,
            height=self.output_height)
        return image

    # ---
    # adapted from  Christophe Ecabert code
    def __find_non_reflective_similarity(
            self,
            sources: torch.Tensor,
            targets: torch.Tensor
            ) -> torch.Tensor:
        K = 2
        M = sources.shape[0]
        x = sources[:, 0:1]
        y = sources[:, 1:2]
        tmp1 = torch.hstack(
            (x,
             y,
             torch.ones((M, 1), device=self.device),
             torch.zeros((M, 1), device=self.device)))
        tmp2 = torch.hstack(
            (y,
             -x,
             torch.zeros((M, 1), device=self.device),
             torch.ones((M, 1), device=self.device)))
        X = torch.vstack((tmp1, tmp2))
        u = targets[:, 0:1]
        v = targets[:, 1:2]
        U = torch.vstack((u, v))
        if torch.linalg.matrix_rank(X) >= 2 * K:
            r, _, _, _ = torch.linalg.lstsq(X, U, rcond=None)
            r = torch.squeeze(r)
        else:
            raise RuntimeError('cp2tform:twoUniquePointsReq')
        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]
        trsfrm = torch.tensor(
            [[sc, -ss, 0],
            [ss,  sc, 0],
            [tx,  ty, 1]],
            device=self.device)
        return trsfrm

    # ---
    # adapted from  Christophe Ecabert code
    def __find_similarity(
            self,
            sources: torch.Tensor,
            targets: torch.Tensor
            ) -> torch.Tensor:
        trans1 = self.__find_non_reflective_similarity(sources, targets)
        points = torch.hstack(
            [sources,
             torch.ones((sources.shape[0], 1), device=self.device)])
        xy = torch.einsum('pi, ij -> pj', points, trans1)
        tgt1 = xy[:, 0:-1]
        norm1 = torch.linalg.norm(tgt1 - targets)
        targets_R = targets.clone()
        targets_R[:, 0] = -1 * targets_R[:, 0]
        trans2r = self.__find_non_reflective_similarity(sources, targets_R)
        trsfrm_reflect_y = torch.tensor(
            [[-1.0, 0.0, 0.0],
             [ 0.0, 1.0, 0.0],
             [ 0.0, 0.0, 1.0]],
            device=self.device)
        trans2 = torch.einsum('ij, jk -> ik', trans2r, trsfrm_reflect_y)
        points = torch.hstack(
            [sources,
             torch.ones((sources.shape[0], 1), device=self.device)])
        xy = torch.einsum('pi, ij -> pj', points, trans2)
        tgt2 = xy[:, 0:-1]
        norm2 = torch.linalg.norm(tgt2 - targets)
        if norm1 <= norm2:
            return trans1
        else:
            return trans2

    # ---

    def __crop_arcface(
            self,
            image : torch.Tensor,
            face_landmarks : LandmarkDetector.FaceLandmarks
            ) -> torch.Tensor:
        batch_size = image.shape[0]
        image_points = torch.stack((
            face_landmarks.eye_left,
            face_landmarks.eye_right,
            face_landmarks.nose,
            face_landmarks.mouth_left,
            face_landmarks.mouth_right),
            dim=1)
        assert image_points.shape[0] == batch_size
        reference_points = torch.tensor([
            self.output_config.eye_left,
            self.output_config.eye_right,
            self.output_config.nose,
            self.output_config.mouth_left,
            self.output_config.mouth_right],
            device=self.device)
        scale = 1.0
        if scale != 1.0:
            center_ref = reference_points.mean(axis=0, keepdims=True)
            ref_pts = center_ref + (reference_points - center_ref) * scale
            reference_points = ref_pts
        # TODO check if similar (should be)
        # transformation = torch.empty(
        #     (batch_size, 3, 3),
        #     device=self.device,
        #     requires_grad=False)
        # for i in range(batch_size):
        #     trsfrm = self.__find_similarity(image_points[0], reference_points)
        #     trsfrm = trsfrm.T
        #     transformation[i] = trsfrm
        # image = kornia.geometry.transform.warp_perspective(
        #     src=image,
        #     M=transformation,
        #     dsize=(self.output_height, self.output_width))
        transformation = torch.empty(
            (batch_size, 2, 3),
            device=self.device,
            requires_grad=False)
        for i in range(batch_size):
            trsfrm = self.__find_similarity(image_points[0], reference_points)
            trsfrm = trsfrm[:, 0:2]
            trsfrm = trsfrm.T
            transformation[i] = trsfrm
        image = kornia.geometry.transform.warp_affine(
            src=image,
            M=transformation,
            dsize=(self.output_height, self.output_width))
        return image

    # ---

    def __resize(
            self,
            image : torch.Tensor,
            width : int,
            height : int,
            mode : str = 'bicubic'
            ) -> torch.Tensor:
        """
            Resize a given image to the desired width and height.
        """
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 4
        return torch.nn.functional.interpolate(
            image,
            mode=mode,
            size=(height, width),
            align_corners=False,
            antialias=True)

    # ---

    def __crop(
            self,
            image : torch.Tensor,
            left : int | torch.Tensor,
            width : int,
            up : int | torch.Tensor,
            height : int
            ) -> torch.Tensor:
        """
            Crop an image and pad it if needed (the bounds can be negative.)
            It is possible to give left and up parameters as 1D torch
            torch.int32 arrays for image batch cropping.
        """
        assert isinstance(image, torch.Tensor)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert image.ndim == 4
        height_0 = image.shape[2]
        width_0 = image.shape[3]
        same_for_all = None
        need_padding = False
        num_frames = image.shape[0]
        if isinstance(left, int) and isinstance(up, int):
            same_for_all = True
        elif type(left) == torch.Tensor and type(up) == torch.Tensor:
            assert left.shape[0] == num_frames
            assert left.dtype == torch.int32
            assert up.shape[0] == num_frames
            assert up.dtype == torch.int32
            same_for_all = False
        else:
            raise RuntimeError('bounds can be either all integers or torch.Tensor.')
        if same_for_all:
            if left < 0 or \
               up < 0 or \
               width + left > width_0 or \
               height + up > height_0:
                need_padding = True
        else:
            if left.amin() < 0 or \
               up.amin() < 0 or \
               width + left.amax() > width_0 or \
               height + up.amax() > height_0:
                need_padding = True
        def __crop(image, left, up):
            image = image[:, :,
                up : up + height,
                left : left + width]
            return image
        def __pad_and_crop(image, left, up, pad_left, pad_right, pad_up, pad_down):
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_up, pad_down))
            image =  image[
                :,
                :,
                up + pad_up: up + pad_up + height,
                left + pad_left: left + pad_left + width]
            return image
        if not need_padding:
            if same_for_all:
                return __crop(image, left, up)
            else:
                return None
        else:
            if same_for_all:
                pad_left = - min(left, 0)
                pad_right = max(width + left - width_0, 0)
                pad_up = - min(up, 0)
                pad_down = max(height + up - height_0, 0)
                return __pad_and_crop(image, left, up, pad_left, pad_right, pad_up, pad_down)
            else:
                return None

# ---

@click.command(
    help='Crop an image to a given specification')
@click.option(
    '--input',
    '-i',
    type=click.Path(exists=True, dir_okay=True),
    help='Input file or directory',
    required=True)
@click.option(
    '--input-config',
    '-ic',
    type=click.Choice(Cropper.get_input_configs()),
    help='Input config',
    required=True)
@click.option(
    '--output',
    '-o',
    type=click.Path(exists=False, dir_okay=True),
    help='Output file or directory',
    required=True)
@click.option(
    '--output-config',
    '-oc',
    type=click.Choice(Cropper.get_output_configs()),
    help='Output config',
    required=True)
@click.option(
    '--device',
    '-d',
    type=click.Choice(['cpu', 'cuda'], case_sensitive=False),
    help='Compute device',
    default='cpu')
def crop(
        input : str,
        input_config : str,
        output : str,
        output_config : str,
        device : str):
    click.echo(f'input   : {input} ({input_config})')
    click.echo(f'output  : {output} ({output_config})')
    click.echo(f'device  : {device}')
    device = torch.device(device=device)
    cropper = Cropper(
        input_config=input_config,
        output_config=output_config,
        device=device)
    image = utils.load_image(input, device=device)
    image = cropper.crop(image)
    utils.save_image(image, output)
