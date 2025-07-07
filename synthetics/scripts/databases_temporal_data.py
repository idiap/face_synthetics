#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import os

import click
import torch

from synthetics import Plot, DatFile

# ---
@click.command(help='Extract temporal stats to .dat format')
@click.option('--input-directory', required=True)
@click.option('--output-directory', required=True)
def main(input_directory : str, output_directory : str) -> None:
    click.echo('Searching for stats files...')
    directories = os.listdir(input_directory)
    names = []
    for directory in directories:
        directory_path = os.path.join(input_directory, directory)
        if not os.path.exists(os.path.join(directory_path, 'samples.h5')):
            continue
        if not os.path.exists(os.path.join(directory_path, 'stats.h5')):
            continue
        name = directory
        if os.path.exists(os.path.join(output_directory, name + '.dat')):
            continue
        click.echo(f'-> {name}')
        names.append(name)
    click.echo('Extracting stats files...')
    plot_engine = Plot()
    for name in names:
        try:
            stats_path = os.path.join(input_directory, name, 'stats.h5')
            click.echo(name)
            data = plot_engine.load_stats_file(stats_name=name, file_path=stats_path)
            num_iterations = data.dist_avg.shape[0]
            iter_num = torch.tensor(range(num_iterations))
            dat_file = DatFile()
            dat_file.add_column(iter_num)      # col 1
            dat_file.add_column(data.time)     # col 2
            dat_file.add_column(data.timestep) # col 3
            dat_file.add_column(data.dist_min) # col 4
            dat_file.add_column(data.dist_avg) # col 5
            dat_file.add_column(data.dist_max) # col 6
            dat_file.add_column(data.w2wd_min) # col 7
            dat_file.add_column(data.w2wd_avg) # col 8
            dat_file.add_column(data.w2wd_max) # col 9
            dat_file.add_column(data.wavg_avg) # col 10
            dat_file.add_column(data.intf_min) # col 11
            dat_file.add_column(data.intf_avg) # col 12
            dat_file.add_column(data.intf_max) # col 13
            dat_file.add_column(data.latf_min) # col 14
            dat_file.add_column(data.latf_avg) # col 15
            dat_file.add_column(data.latf_max) # col 16
            dat_file.add_column(data.rndf_min) # col 17
            dat_file.add_column(data.rndf_avg) # col 18
            dat_file.add_column(data.rndf_max) # col 19
            dat_file.add_column(data.totf_min) # col 20
            dat_file.add_column(data.totf_avg) # col 21
            dat_file.add_column(data.totf_max) # col 22
            data.prop_ict = 1.0 - data.prop_ict
            dat_file.add_column(data.prop_ict) # col 23
            output_path = os.path.join(output_directory, name + '.dat')
            dat_file.export(file_path=output_path)
            click.echo(output_path)
        except:
            click.echo('error')
            pass

# ---
if __name__ == '__main__':
    main()

# ---
    