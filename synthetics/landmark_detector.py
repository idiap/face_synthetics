#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import enum
from dataclasses import dataclass

import click

import torch
import numpy as np
import dlib
from PIL import Image
from facenet_pytorch import MTCNN
import kornia

from . import utils

# ---

class LandmarkDetector():

    @dataclass
    class FaceLandmarks:
        landmarks_68p : torch.Tensor | None = None
        eye_left : torch.Tensor | None = None
        eye_right : torch.Tensor | None = None
        nose : torch.Tensor | None = None
        mouth_left : torch.Tensor | None = None
        mouth_right : torch.Tensor | None = None
        error : list[bool] | None = None
        num_images : int = 0

    # ---

    @enum.unique
    class Backend(enum.Enum):
        MTCNN = 1
        DLIB = 2
        KORNIA = 3

    # ---

    def __init__(
            self,
            backend : Backend = Backend.MTCNN,
            device = torch.device('cuda'),
            dtype = torch.float32) -> None:
        self.backend : self.Backend = backend
        self.device = device
        self.dtype = dtype
        if self.backend == self.Backend.MTCNN:
            self.mtccn_detector = MTCNN(device=device)
        elif self.backend == self.Backend.DLIB:
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(
                utils.get_model_path('dlib_lmd'))
        elif self.backend == self.Backend.KORNIA:
            self.kornia_face_detector = kornia.contrib.FaceDetector()
            self.kornia_face_detector.to(self.device)
        else:
            raise RuntimeError('Unsupported backend')

    # ---
    # adapted from deep_3dfr/util/preprocessing.py:extract_5p()
    def __get_main_keypoints_from_68p(
            self,
            landmarks : FaceLandmarks
            ) -> FaceLandmarks:
        """ Compute the 5 points landmarks from the 68p ones. """
        assert torch.is_tensor(landmarks.landmarks_68p)
        assert landmarks.landmarks_68p.ndim == 3
        assert landmarks.landmarks_68p.shape[1] == 68
        assert landmarks.landmarks_68p.shape[2] == 2
        lm_idx = torch.tensor([31, 37, 40, 43, 46, 49, 55]) - 1
        landmarks.eye_left = torch.mean(
            landmarks.landmarks_68p[:, lm_idx[[3, 4]], :], 1)
        landmarks.eye_right = torch.mean(
            landmarks.landmarks_68p[:, lm_idx[[1, 2]], :], 1)
        landmarks.nose = landmarks.landmarks_68p[:, lm_idx[0], :]
        landmarks.mouth_left = landmarks.landmarks_68p[:, lm_idx[6], :]
        landmarks.mouth_right = landmarks.landmarks_68p[:, lm_idx[5], :]
        return landmarks

    # ---

    def __detect_mtcnn(
            self,
            image : torch.Tensor
            ) -> FaceLandmarks:
        num_images : int = image.shape[0]
        landmarks = self.FaceLandmarks()
        landmarks.num_images = num_images
        landmarks.error = []
        landmarks.eye_left = torch.empty((num_images, 2), device=self.device)
        landmarks.eye_right = torch.empty((num_images, 2), device=self.device)
        landmarks.mouth_left = torch.empty((num_images, 2), device=self.device)
        landmarks.mouth_right = torch.empty((num_images, 2), device=self.device)
        landmarks.nose = torch.empty((num_images, 2), device=self.device)
        image_np : np.ndarray = utils.image_to_numpy(image=image)
        # TODO import model backbone and vectorize there
        # TODO report detection error here?
        for i in range(num_images):
            image_pil : Image = utils.numpy_to_pil(image=image_np[i])
            _, _, landmarks_5p = self.mtccn_detector.detect(image_pil, landmarks=True)
            if landmarks_5p is not None:
                landmarks_5p = landmarks_5p.astype(np.float32)
                landmarks_5p = landmarks_5p[0]
                landmarks_5p = torch.tensor(landmarks_5p, device=self.device)
                landmarks.eye_right[i, :] = landmarks_5p[0, :] # eye_left
                landmarks.eye_left[i, :] = landmarks_5p[1, :] # eye_right
                landmarks.nose[i, :] = landmarks_5p[2, :] # nose
                landmarks.mouth_right[i, :] = landmarks_5p[3, :] # mouth_left
                landmarks.mouth_left[i, :] = landmarks_5p[4, :] # mouth_right
                landmarks.error.append(False)
            else:
                landmarks.error.append(True)
        return landmarks

    # ---

    def __detect_dlib(
            self,
            image : torch.Tensor
            ) -> FaceLandmarks:
        num_images = image.shape[0]
        face_landmarks = self.FaceLandmarks()
        face_landmarks.num_images = num_images
        face_landmarks.error = []
        face_landmarks.landmarks_68p = torch.zeros(
            [num_images, 68, 2], 
            device=self.device,
            dtype=self.dtype)
        image = utils.image_to_numpy(image)
        image = utils.numpy_to_matplotlib(image)
        for index in range(num_images):
            img = image[index]
            detection_list = self.dlib_detector(img, 1)
            error = False
            try:
                detection = detection_list[0]
            except:
                error = True
            if not error:
                shape = self.dlib_predictor(img, detection)
                face_landmarks.landmarks_68p[index, :, :] = torch.tensor(
                    [(item.x, item.y) for item in shape.parts()], 
                    device=self.device,
                    dtype=self.dtype)
            face_landmarks.error.append(error)
        face_landmarks = self.__get_main_keypoints_from_68p(face_landmarks)
        return face_landmarks

    # ---

    def __detect_kornia(
            self,
            image : torch.Tensor
            ) -> FaceLandmarks:
        num_images : int = image.shape[0]
        image = utils.adjust_dynamic_range(
            image=image,
            input_range_min=-1.0,
            input_range_max=1.0,
            target_range_min=1.0,
            target_range_max=255.0)
        results : list[torch.Tensor] = self.kornia_face_detector(image) # B x (N, 15)
        assert len(results) == num_images
        assert num_images == 1
        error = False
        try:
            result = results[0] # TODO batch (are we sure here ?)
            result = result[0] # keep first box
        except:
            error = True
        face_landmarks = self.FaceLandmarks()
        face_landmarks.num_images = num_images
        face_landmarks.error = []
        face_landmarks.eye_left = torch.empty((num_images, 2), device=self.device)
        face_landmarks.eye_right = torch.empty((num_images, 2), device=self.device)
        face_landmarks.nose = torch.empty((num_images, 2), device=self.device)
        face_landmarks.mouth_left = torch.empty((num_images, 2), device=self.device)
        face_landmarks.mouth_right = torch.empty((num_images, 2), device=self.device)
        face_landmarks.eye_right[0, :] = result[[4, 5]]
        face_landmarks.eye_left[0, :] = result[[6, 7]]
        face_landmarks.mouth_right[0, :] = result[[8, 9]]
        face_landmarks.mouth_left[0, :] = result[[10, 11]]
        face_landmarks.nose[0, :] = result[[12, 13]]
        face_landmarks.error.append(error)
        return face_landmarks

    # ---

    def detect(
            self, 
            image : torch.Tensor
            ) -> FaceLandmarks:
        """ Run a landmark detector on the input image (batch).
            Image format (b c y x, float[-1.0, 1.0]). """
        assert torch.is_tensor(image)
        assert image.ndim == 4
        if self.backend == self.Backend.MTCNN:
            return self.__detect_mtcnn(image=image)
        elif self.backend == self.Backend.DLIB:
            return self.__detect_dlib(image=image)
        elif self.backend == self.Backend.KORNIA:
            return self.__detect_kornia(image=image)
        else:
            raise RuntimeError('Unknown backend')
    
    # ---

    def draw(
            self,
            image : torch.Tensor, 
            landmarks : FaceLandmarks
            ) -> None:
        """ Draw landmarks on an image. """
        assert torch.is_tensor(image)
        assert image.ndim == 4
        num_images = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]
        def draw_dot(
                image : torch.Tensor,
                index : int,
                pos: torch.Tensor,
                color : torch.Tensor,
                size : int
                ) -> None:
            x = int(pos[0])
            y = int(pos[1])
            for i_x in range(-size, size):
                for i_y in range(-size, size):
                    if  y + i_y >= 0 and \
                        y + i_y < height and \
                        x + i_x >= 0 and \
                        x + i_y < width:
                            image[index, :, y + i_y, x + i_x] = color
            return image
        green = torch.tensor([-1.0, 1.0, -1.0])
        blue = torch.tensor([-1.0, -1.0, 1.0])
        for i in range(num_images):
            if landmarks.landmarks_68p is not None:
                for j in range(68):
                    image=draw_dot(
                        image=image,
                        index=i,
                        pos=landmarks.landmarks_68p[i, j, :],
                        color=blue,
                        size=2)
            if landmarks.eye_left is not None:
                image = draw_dot(
                    image=image,
                    index=i,
                    pos=landmarks.eye_left[i],
                    color=green,
                    size=2)
            if landmarks.eye_right is not None:
                image = draw_dot(
                    image=image,
                    index=i,
                    pos=landmarks.eye_right[i],
                    color=green,
                    size=2)
            if landmarks.nose is not None:
                image = draw_dot(
                    image=image,
                    index=i,
                    pos=landmarks.nose[i],
                    color=green,
                    size=2)
            if landmarks.mouth_left is not None:
                image = draw_dot(
                    image=image,
                    index=i,
                    pos=landmarks.mouth_left[i],
                    color=green,
                    size=2)
            if landmarks.mouth_right is not None:
                image = draw_dot(
                    image=image,
                    index=i,
                    pos=landmarks.mouth_right[i],
                    color=green,
                    size=2)
        return image

# ---

@click.command(
    help='Detect face landmarks from an image')
@click.option(
    '--input', 
    '-i', 
    type=click.Path(exists=True, dir_okay=False), 
    help='Input image file', 
    required=True)
@click.option(
    '--backend', 
    '-b', 
    type=click.Choice(['mtcnn', 'dlib', 'kornia']), 
    help='Backend used to perform the detection', 
    default='mtcnn')
@click.option(
    '--device', 
    '-d', 
    type=click.Choice(['cpu', 'cuda']), 
    help='Backend used to perform the detection', 
    default='cpu')
@click.option(
    '--output', 
    '-o', 
    type=click.Path(exists=False, dir_okay=False), 
    help='Output image file with landmarks (optional)', 
    required=False)
def detect_landmarks(
        input : str,
        backend : str,
        device : str,
        output : str | None
        ) -> None:
    device : torch.device = torch.device(device)
    if backend == 'mtcnn':
        backend = LandmarkDetector.Backend.MTCNN
    elif backend == 'dlib':
        backend = LandmarkDetector.Backend.DLIB
    elif backend == 'kornia':
        backend = LandmarkDetector.Backend.KORNIA
    else:
        raise RuntimeError('Unknown backend')
    image = utils.load_image(
        file_path=input,
        device=device)
    landmark_detector = LandmarkDetector(
        backend=backend,
        device=device)
    landmarks = landmark_detector.detect(image=image)
    print(landmarks)
    if output is not None:
        image = landmark_detector.draw(
            image=image,
            landmarks=landmarks)
        utils.save_image(image=image, file_path=output)
        