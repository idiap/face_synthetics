#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#

import click
import torch

from synthetics import Plot, DatFile

@click.command(help='Extract w2wd from multiple databases stats')
@click.option('--add-stats-file', 'files', required=True, multiple=True)
@click.option('--stats-file-num-ids', 'num_ids', required=True, multiple=True)
@click.option('--add-iteration', 'iterations', required=True, multiple=True)
@click.option('--output-file', required=True)
def main(
        files : tuple[str],
        num_ids : tuple[int],
        iterations: tuple[int],
        output_file : str
        ) -> None:
    click.echo('Openning stats files...')
    assert len(files) == len(num_ids)
    num_files = len(files)
    columns = []
    iterations = [int(iteration) for iteration in iterations]
    num_ids = [int(num_id) for num_id in num_ids]
    for _ in iterations:
        column = torch.empty((num_files, ), dtype=torch.float)
        columns.append(column)
    plot_engine = Plot()
    for i, file in enumerate(files):
        click.echo(f'-> {file}')
        data = plot_engine.load_stats_file(stats_name=file, file_path=file)
        data_w2w = data.w2wd_avg
        for col, iteration in enumerate(iterations):
            columns[col][i] = data_w2w[iteration]
    click.echo('Writing dat file...')
    dat_file = DatFile()
    num_ids = torch.tensor(num_ids)
    dat_file.add_column(num_ids)
    for column in columns:
        dat_file.add_column(column)
    dat_file.export(file_path=output_file)

if __name__ == '__main__':
    main()
    