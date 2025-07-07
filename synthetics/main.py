#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import time
import click

from .cropper import crop
from .generator import generate
from .embedding import get_embedding, get_database_embeddings
from .plot import plot
from .projector import project
from .project_database import project_database
from .landmark_detector import detect_landmarks
from .face_extractor_3d import extract_3d_face
from .latent_analysis import latent_analysis
from .latent_edit import latent_edit
from .database_generator import generate_database

start_time = None

def greetings():
    click.echo("*******************************************************************************")
    click.echo("                                 _   _          _   _                          ")
    click.echo("                                | | | |        | | (_)                         ")
    click.echo("                 ___ _   _ _ __ | |_| |__   ___| |_ _  ___ ___                 ")
    click.echo("                / __| | | | '_ \| __| '_ \ / _ \ __| |/ __/ __|                ")
    click.echo("                \__ \ |_| | | | | |_| | | |  __/ |_| | (__\__ \                ")
    click.echo("                |___/\__, |_| |_|\__|_| |_|\___|\__|_|\___|___/                ")
    click.echo("                     __/ /                                                     ")
    click.echo("                    |___/                                                      ")
    click.echo("*******************************************************************************")

@click.group()
@click.pass_context
def cli(context : click.core.Context):
    global start_time
    start_time = time.time()

    greetings()
    if context.invoked_subcommand is not None:
        click.echo(f'command : {context.invoked_subcommand}')

    context.call_on_close(_on_close)

def _on_close():
    global start_time
    click.echo("*******************************************************************************")
    click.echo(f"> total time: {time.time() - start_time}")
    click.echo("*******************************************************************************")

cli.add_command(crop)
cli.add_command(generate)
cli.add_command(get_embedding)
cli.add_command(get_database_embeddings)
cli.add_command(project)
cli.add_command(project_database)
cli.add_command(detect_landmarks)
cli.add_command(extract_3d_face)
cli.add_command(latent_analysis)
cli.add_command(latent_edit)
cli.add_command(generate_database)
cli.add_command(plot)

if __name__ == '__main__':
    cli()
