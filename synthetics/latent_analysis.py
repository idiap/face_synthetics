#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# Adapted from:
# https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_synthetic_dataset/
# This code was entirely written by a human

import os
import pickle

import click
from tqdm import tqdm

import numpy as np
import torch
import pandas as pd
from sklearn.svm import LinearSVC

from . import Database, MultipieDatabase
from . import utils
from .utils import Sample

# ---

class LatentAnalyzer:

    VALID_COVARIATES = ['illumination','expression','pose']

    POS_TO_CAM = \
    {
        'frontal': ['05_1'],
        'left': ['11_0', '12_0', '09_0', '08_0', '13_0', '14_0'],
        'right': ['05_0', '04_1', '19_0', '20_0', '01_0', '24_0'],
    }

    CAM_TO_POS = \
    {
        cam: pos for pos, cam_list in POS_TO_CAM.items() for cam in cam_list
    }

    POS_TO_FLASH = \
    {
        'no_flash': [0],
        'left': [1,2,3],
        'left_discard' : list(range(4, 7)) + [14, 15],
        'frontal': [7, 16],
        'right': [11, 12, 13],
        'right_discard': list(range(8, 11)) + [17, 18],
    }

    FLASH_TO_POS = \
    {
        flash: pos for pos, flash_list in POS_TO_FLASH.items() for flash in flash_list
    }

    # Recording = SessionNumber_RecordingNumber
    EXPRESSION_TO_RECORDING = \
    {
        'neutral': ['01_01', '02_01', '03_01', '04_01', '04_02'],
        'smile': ['01_02', '03_02'],
        'surprise': ['02_02'],
        'squint': ['02_03'],
        'disgust': ['03_03'],
        'scream': ['04_03']
    }

    RECORDING_TO_EXPRESSION = \
    {
        recording : expr for expr, recording_list in EXPRESSION_TO_RECORDING.items() for recording in recording_list
    }

    # ---

    def __init__(
            self,
            network_type : str,
            projected_database_directory : str
            ) -> None:
        self.network_type = network_type
        self.projected_database_directory = projected_database_directory # MAIN_DIR
        self.latent_directory = os.path.join(self.projected_database_directory, 'projected') # LAT_DIR
        self.failure_file = os.path.join(self.projected_database_directory, 'failure.dat') # FAILURE_FILE

    # ---

    def filter_failure_cases(
            self,
            samples : list[Database.Sample]
            ) -> list[Database.Sample]:
        with open(self.failure_file, 'r') as f:
            failures = [item.rstrip() for item in f.readlines()]
        return [sample for sample in samples if sample.key not in failures]

    # ---

    def get_covariate(
            self,
            sample : Database.Sample,
            covariate : str
            ) -> str:
        if covariate == 'expression':
            recording = '_'.join(sample.key.split('_')[7:9])
            return self.RECORDING_TO_EXPRESSION[recording]
        elif covariate == 'pose':
            camera = '_'.join(sample.key.split('_')[4:6])
            return self.CAM_TO_POS[camera]
        elif covariate == 'illumination':
            shot = sample.key.split('_')[-1]
            return self.FLASH_TO_POS[int(shot)]
        else:
            raise ValueError(
                'Unknown covariate {} not in {}'.format(covariate, self.VALID_COVARIATES))

    # ---

    def get_sample_list(
            self,
            covariate : str,
            group : str = 'train'
            ) -> list[Database.Sample]:
        ''' Get the list of databases samples to be processed. '''
        database : Database = MultipieDatabase()
        if group == 'all':
            groups = None
        else:
            groups = [group]
        if covariate == 'illumination':
            samples = database.query(protocol_names=['U'], group_names=groups)
        elif covariate == 'expression':
            samples = database.query(protocol_names=['E_lit'], group_names=groups)
        elif covariate == 'pose':
            samples = database.query(protocol_names=['P_center_lit'], group_names=groups)
        else:
            raise ValueError('Unknown covariate {}'.format(covariate))
        return self.filter_failure_cases(samples)

    # ---

    def load_latents(
            self,
            covariate : str,
            group : str = 'train'
            ) -> pd.DataFrame:
        assert covariate in self.VALID_COVARIATES, 'Unknown `covariate` {} not in {}'.format(
            covariate, self.VALID_COVARIATES)
        database_samples = self.get_sample_list(covariate, group)
        click.echo('Loading latents...')
        df = pd.DataFrame()
        df['file'] = [database_sample.key for database_sample in database_samples]
        def load_sample_latent(database_sample : Database.Sample) -> np.ndarray:
            image_sample_rel_path = database_sample.path
            latent_sample_rel_path = os.path.splitext(image_sample_rel_path)[0] + '.h5'
            latent_sample_full_path = os.path.join(self.latent_directory, latent_sample_rel_path)
            sample : Sample = Sample()
            sample.load(file_path=latent_sample_full_path, device=torch.device('cpu'))
            w_latent : torch.Tensor = sample.w_latent
            w_latent : np.ndarray = sample.w_latent.numpy()[0]
            return w_latent
        df['latent'] = [load_sample_latent(database_sample) for database_sample in tqdm(database_samples)]
        df[covariate] = [self.get_covariate(database_sample, covariate) for database_sample in tqdm(database_samples)]
        click.echo('... done.')
        return df

    # ---

    def binary_analysis(
            self,
            train_df : pd.DataFrame,
            covariate  : str,
            target_labels : list[str],
            seed : int | None = None,
            **kwargs
            ) -> dict:
        train_df = train_df[train_df[covariate].isin(target_labels)]
        svm = LinearSVC(
            fit_intercept=False,
            random_state=seed,
            max_iter=10000,
            dual='auto',
            **kwargs)
        train_latents = np.stack(train_df['latent'])
        train_labels = np.stack(train_df[covariate])
        # Ensure that 0 and 1 classes match the order of the target_labels
        train_labels = np.where(train_labels == target_labels[0], 0, 1)
        svm.fit(train_latents, train_labels)
        normal = svm.coef_ / np.linalg.norm(svm.coef_)
        distances = train_latents.dot(normal.T)
        negatives = distances[distances < 0]
        positives = distances[distances > 0]
        def stats(distances):
            return \
            {
                'min': np.min(distances),
                'max': np.max(distances),
                'mean': np.mean(distances),
                'std': np.std(distances),
            }
        neg_stats = stats(negatives)
        pos_stats = stats(positives)
        return \
        {
            'svm': svm,
            'normal': normal,
            'neg_stats': neg_stats,
            'pos_stats': pos_stats,
        }

    # ---

    def multiclass_analysis(
            self,
            train_df : pd.DataFrame,
            covariate : str,
            neutral_label : str,
            other_labels : list[str],
            **kwargs):
        return \
        {
            (neutral_label, label): self.binary_analysis(
                train_df,
                covariate=covariate,
                target_labels=[neutral_label, label],
                **kwargs)
            for label in other_labels
        }

    # ---

    def pose_analysis(
            self,
            train_df : pd.DataFrame,
            seed : int | None = None
            ) -> dict:
        return self.binary_analysis(
            train_df=train_df,
            covariate='pose',
            target_labels=['left', 'right'],
            seed=seed)

    # ---

    def illumination_analysis(
            self,
            train_df : pd.DataFrame,
            seed : int | None = None
            ) -> dict:
        return self.binary_analysis(
            train_df=train_df,
            covariate='illumination',
            target_labels=['left', 'right'],
            seed=seed)

    # ---

    def expression_analysis(
            self,
            train_df : pd.DataFrame,
            seed : int | None = None
            ) -> dict:
        return self.multiclass_analysis(
            train_df=train_df,
            covariate='expression',
            neutral_label='neutral',
            other_labels=['smile', 'scream', 'disgust', 'squint', 'surprise'],
            seed=seed)

    # ---

    def covariate_analysis(
            self,
            covariate : str,
            train_df : pd.DataFrame,
            seed : int | None = None):
        if covariate == 'expression':
            return self.expression_analysis(train_df, seed)
        elif covariate == 'pose':
            return self.pose_analysis(train_df, seed)
        elif covariate == 'illumination':
            return self.illumination_analysis(train_df, seed)
        else:
            raise ValueError(
                'Unknown covariate {} not in {}'.format(covariate, self.VALID_COVARIATES))

# ---

@click.command(
    help='Compute and save latent directions starting from precomputed latent projections of MultiPIE')
@click.pass_context
@click.option(
    '--projections-dir',
    '-i',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    help='Root of directory containing Multipie projections',
    required=True)
@click.option(
    '--output-path',
    '-o',
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help='Path where to store the latent directions (pickle file)',
    required=True)
@click.option(
    '--force',
    '-f',
    is_flag=True,
    help='Activate flag to overwrite computed directions if the file already exist')
@click.option(
    '--no-expression',
    is_flag=True,
    help='Deactivate expression SVM fitting.')
@click.option(
    '--no-pose',
    is_flag=True,
    help='Deactivate pose SVM fitting.')
@click.option(
    '--no-illumination',
    is_flag=True,
    help='Deactivate illumination SVM fitting.')
@click.option(
    '--seed',
    '-s',
    type=int,
    help='Seed to control stochasticity during the SVM fitting.',
    default=None)
def latent_analysis(
        ctx : click.Context,
        projections_dir : str,
        output_path : str,
        force : bool,
        no_expression : bool,
        no_pose : bool,
        no_illumination : bool,
        seed : int
        ) -> None:
    latent_analyzer = LatentAnalyzer(
        network_type=None,
        projected_database_directory=projections_dir)
    covariates = []
    if not no_illumination:
        covariates.append('illumination')
    if not no_expression:
        covariates.append('expression')
    if not no_pose:
        covariates.append('pose')
    if (not os.path.exists(output_path)) or force:
        out = {}
        for covariate in covariates:
            click.echo('Analyzing {} covariate ...'.format(covariate))
            df = latent_analyzer.load_latents(covariate)
            click.echo('SVM fitting ...')
            result = latent_analyzer.covariate_analysis(covariate, df, seed)
            out[covariate] = result
        with open(output_path, 'wb') as f:
            pickle.dump(out, f)
    else:
        click.echo('Computed directions are already found under {}. Use the --force flag to overwrite them.'.format(output_path))
