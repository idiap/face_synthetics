#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import os
import click

import torch

from . import utils
from .utils import Sample
from .cropper import Cropper
from .face_extractor_3d import FaceExtractor3D
from .generator import Generator
from .projector import Projector
from .databases import MultipieDatabase, Database

# ---

CROP_DIR = "img_aligned"
PROJ_DIR = "projected"

def is_processed(
        sample : MultipieDatabase.Sample,
        output_dir : str
        ) -> bool:
    database_sample_path = sample.path
    projected_sample_path = os.path.splitext(database_sample_path)[0] + '.h5'
    projected_sample_path_full = os.path.join(output_dir, PROJ_DIR, projected_sample_path)
    return os.path.exists(projected_sample_path_full)

# ---

@click.command(help="Project a database into a GAN latent space")
@click.pass_context
@click.option(  "--database",
                "-d",
                type=click.Choice(['multipie']),
                help="Which database to project (available : 'multipie')",
                default='multipie')
@click.option(  "--protocol",
                "-p",
                type=click.Choice(['U', 'P_center_lit', 'E_lit']),
                help="Which database protocol to project (available : 'U', 'E_lit' and 'P_center_lit'), default: all three",
                default=None)
@click.option(  "--group",
                "-g",
                type=click.Choice(['train', 'dev', 'eval', 'all']),
                help="Either `train`, `dev` or `eval` or all together `all`. Default: `train`",
                default='train')
@click.option(  "--output-dir",
                "-o",
                required=True,
                help="Root directory of the output files")
@click.option(  '--network-type',
                '-nt', help='Network',
                type=click.Choice(utils.network_types()),
                default='stylegan2')
@click.option(  "--num_steps",
                "-n",
                type=int,
                default=1000,
                help="Number of projection steps")
@click.option(  "--checkpoint",
                "-c",
                is_flag=True,
                help="Activate flag to checkpoint cropped faces and projected faces")
@click.option(  "--force",
                "-f",
                is_flag=True,
                help="Activate flag to overwrite computed latent if they already exist")
@click.option(  "--dry-run",
                is_flag=True,
                help="Do not perform the projections, just count remaining samples")
@click.option(  "--seed",
                "-s",
                type=int,
                default=None,
                help="Seed to control stochasticity during projection")
def project_database(
        ctx : click.Context,
        database : str,
        protocol : str,
        group : str,
        output_dir : str,
        network_type : str,
        num_steps : int,
        checkpoint : bool,
        force : bool,
        dry_run : bool,
        seed):

    # Parallelization
    task_id, num_tasks = utils.get_task()
    failure_file_path = os.path.join(output_dir, "failure.dat")

    # Load projector and croppers
    device = torch.device('cuda:0')
    if network_type == 'eg3d':
        cropper_config = Cropper.Config.eg3d
    elif network_type == 'stylegan2-256':
        cropper_config = Cropper.Config.ffhq256
    else:
        cropper_config = Cropper.Config.ffhq
    cropper = Cropper(
        input_config=Cropper.Config.dlib,
        output_config=cropper_config,
        device=device)
    projector_parameters = Projector.ProjectorParameters(
        num_steps=num_steps,
        projection_seed=seed,
        truncation_psi=0.7,
        start_wavg=True)
    projector = Projector(
        network_type=network_type,
        device=device,
        parameters=projector_parameters)
    generator = Generator(
        network_type=network_type,
        device=device)
    if network_type == 'eg3d':
        face_extractor = FaceExtractor3D(device=device)
        face_extractor_cropper = Cropper(
            input_config=Cropper.Config.dlib,
            output_config=Cropper.Config.deep3dfr,
            device=device)

    # Load database
    if database != 'multipie':
        raise RuntimeError('Unknown database')
    database : Database = MultipieDatabase()
    protocols = [protocol] if protocol is not None else ['U', 'P_center_lit', 'E_lit']
    groups = database.list_protocols() if group == 'all' else [group]
    samples : list[Database.Sample] = database.query(
        protocol_names=protocols,
        group_names=groups)
    if not force:
        samples = [sample for sample in samples if not is_processed(sample, output_dir)]
        if os.path.exists(failure_file_path):
            with open(failure_file_path, "r") as failure_file:
                fail_cases = [item.rstrip() for item in failure_file]
            samples = [sample for sample in samples if sample.key not in fail_cases]
    subsamples : list[Database.Sample] = samples[task_id :: num_tasks]
    print("{} samples remaining. Handling {} of them".format(len(samples), len(subsamples)))
    if dry_run:
        exit()

    # Loop over samples
    for i, sample in enumerate(subsamples):
        # Get image and crop
        print("{} : {}".format(i, sample.key))
        image = database.load_sample(
            sample=sample,
            device=device)
        try:
            cropped = cropper.crop(image)
        except:
            with open(failure_file_path, "a") as failure_file:
                failure_file.write(sample.key + "\n")
            print("Failure to crop {} !".format(sample.key))
            continue
        if checkpoint:
            file_path = os.path.splitext(sample.path)[0]
            file_path = os.path.join(output_dir, CROP_DIR, file_path) + '.png'
            utils.save_image(
                image=cropped,
                file_path=file_path,
                create_directories=True)

        # Get target pose for EG3D
        if network_type == 'eg3d':
            try:
                target_3dfr = face_extractor_cropper.crop(cropped)
            except:
                with open(failure_file_path, "a") as failure_file:
                    failure_file.write(sample.key + "\n")
                print("Failure to crop {} !".format(sample.key))
                continue
            face_extractor.extract(image=target_3dfr)
            # target_depth_map = face_extractor.get_depth_map()
            target_pose = face_extractor.compute_pose()['label']
        else:
            # target_depth_map = None
            target_pose = None

        # Project sample
        projected_w_steps = projector.project(
            target=cropped,
            target_pose=target_pose)
        projected_w = projected_w_steps[-1][0][0].unsqueeze(0)

        # Save results
        synth_image = generator.synthesis(w=projected_w, c=target_pose)
        file_path_no_ext = os.path.splitext(sample.path)[0]
        file_path_no_ext = os.path.join(output_dir, PROJ_DIR, file_path_no_ext)
        if checkpoint:
            utils.save_image(
                image=synth_image,
                file_path=file_path_no_ext + '.png',
                create_directories=True)
        w_sample = Sample(w_latent=projected_w, c_label=target_pose, network_type=network_type)
        w_sample_path = file_path_no_ext + '.h5'
        w_sample.save(
            file_path=w_sample_path,
            create_directories=True)
