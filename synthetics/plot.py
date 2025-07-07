#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, Idiap Research Institute. All rights reserved.
# SPDX-License-Identifier: LicenseRef-IdiapNCResearchAndEducationalOnly
#
# This code was entirely written by a human

import math
from dataclasses import dataclass
from enum import Enum, unique

import click
from tqdm import tqdm
import h5py
import csv
import torch
import einops

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from .utils import Sample, SampleCollection
from .embedding import Embedding

# ---

@dataclass
class SampleCollectionData:
    identities : torch.Tensor | None = None
    latents : torch.Tensor | None = None
    embeddings : torch.Tensor | None = None

@dataclass
class StatsFileData:
    num_iterations : int = 0
    dist_min : torch.Tensor | None = None
    dist_avg : torch.Tensor | None = None
    dist_max : torch.Tensor | None = None
    prop_ict : torch.Tensor | None = None
    w2wd_min : torch.Tensor | None = None
    w2wd_avg : torch.Tensor | None = None
    w2wd_max : torch.Tensor | None = None
    wavg_avg : torch.Tensor | None = None
    intf_min : torch.Tensor | None = None
    intf_avg : torch.Tensor | None = None
    intf_max : torch.Tensor | None = None
    rndf_min : torch.Tensor | None = None
    rndf_avg : torch.Tensor | None = None
    rndf_max : torch.Tensor | None = None
    latf_min : torch.Tensor | None = None
    latf_avg : torch.Tensor | None = None
    latf_max : torch.Tensor | None = None
    totf_min : torch.Tensor | None = None
    totf_avg : torch.Tensor | None = None
    totf_max : torch.Tensor | None = None
    timestep : torch.Tensor | None = None
    time : torch.Tensor | None = None

@dataclass
class TimingFileData:
    num_iterations : int = 0
    num_ids : torch.Tensor | None = None
    wall_time : torch.Tensor | None = None

@unique
class HistogramType(Enum):
    embedding_intra_class = 1
    embedding_inter_class = 2
    latent_intra_class = 3
    latent_inter_class = 4

@unique
class ClassType(Enum):
    intra = 1
    inter = 2
    cross = 3

@unique
class DataType(Enum):
    embedding = 1
    latent = 2

@dataclass
class Histogram:
    class_type : ClassType | None = None
    data_type : DataType | None = None
    num_samples : int | None = None
    bin_values : torch.Tensor | None = None
    bin_count : torch.Tensor | None = None

@dataclass
class PlotCurve:
    x_values : torch.Tensor = None
    y_values : torch.Tensor = None

@dataclass
class PlotAxis:
    label : str = ''
    value_min : float | None = None
    value_max : float | None = None
    log_axis : bool = False
    log_base : float = 10.0

@dataclass
class PlotItem:
    title : str | None = None
    curves : list[PlotCurve] = None
    x_axis : PlotAxis = PlotAxis()
    y_axis : PlotAxis = PlotAxis()

# ---

class DatFile:
    """ Export a .dat file compatible with gnuplot. """

    def __init__(
            self,
            dtype : torch.dtype = torch.float32
            ) -> None:
        self.columns : list[torch.Tensor] = []
        self.dtype = dtype

    def add_column(
            self,
            column : torch.Tensor,

            ) -> None:
        assert column.ndim == 1
        if len(self.columns) > 0:
            assert column.shape[0] == self.columns[0].shape[0]
        column = column.to(self.dtype)
        self.columns.append(column)

    def export(
            self,
            file_path : str
            ) -> None:
        num_lines = self.columns[0].shape[0]
        with open(file=file_path, mode='w') as output:
            for row_number in range(num_lines):
                row : str = ''
                for column in self.columns:
                    row += f'{column[row_number]:16f}'
                output.write(row + '\n')

# ---

class Plot():
    def __init__(
            self,
            device : torch.device = torch.device('cpu'),
            dtype : torch.dtype = torch.float32,
            embedding_type : str = 'iresnet50',
            batch_size : int = 4096
            ) -> None:
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.embedding = Embedding(
            model_name=embedding_type,
            device=self.device,
            dtype=self.dtype)
        self.sample_collections_data = {}
        self.stats_files_data = {}
        self.timing_files_data = {}
        self.histograms = {}
        self.plot_items = {}

    # ---

    def load_sample_collection(
            self,
            collection_name : str,
            file_path : str
            ) -> SampleCollectionData:
        """ Load a sample collection. """
        assert collection_name is not None
        assert collection_name not in self.sample_collections_data.keys(), \
            'collection_name already exists'
        click.echo(f'Loading collection: {collection_name} : {file_path}...')
        sample_collection = SampleCollection(
            file_path=file_path,
            read_only=True)
        sample_collection.load(
            device=self.device,
            dtype=self.dtype)
        identities = sample_collection.list_identities()
        num_identities = len(identities)
        click.echo(f'Found {num_identities} identities...')
        samples : list[Sample] = []
        samples_identities : list[int] = []
        max_labels_per_identity = 0
        for id in tqdm(identities):
            labels = sample_collection.list_identity_labels(identity=id)
            for label in labels:
                sample = sample_collection.get_sample(id, label)
                assert sample.embedding is not None, f'Sample {id} {label} has no embedding'
                samples.append(sample)
                samples_identities.append(id)
            max_labels_per_identity = max(max_labels_per_identity, len(labels))
        num_samples = len(samples)
        click.echo(f'Loading {num_samples} samples...')
        click.echo(f'Maximum labels per identity: {max_labels_per_identity}')
        samples_identities : torch.Tensor = torch.tensor(
            samples_identities,
            device=self.device,
            dtype=torch.int64)
        assert num_samples == samples_identities.shape[0], 'internal error'
        latents = torch.empty(
            (num_samples, self.embedding.e_dim),
            device=self.device,
            dtype=self.dtype)
        embeddings = torch.empty(
            (num_samples, self.embedding.e_dim),
            device=self.device,
            dtype=self.dtype)
        for i, sample in enumerate(samples):
            embeddings[i, :] = sample.embedding[0, :]
        sample_collection_data = SampleCollectionData()
        sample_collection_data.identities = samples_identities
        sample_collection_data.latents = latents
        sample_collection_data.embeddings = embeddings
        self.sample_collections_data[collection_name] = sample_collection_data
        return sample_collection_data

    # ---

    def load_stats_file(
            self,
            stats_name : str,
            file_path : str
            ) -> StatsFileData:
        """ Load a stats file. """
        assert stats_name is not None
        assert stats_name not in self.stats_files_data.keys(), \
            'stats_name already exists'
        click.echo(f'Loading stats file: {stats_name} : {file_path}...')
        stats_file_data = StatsFileData()
        with h5py.File(name=file_path, mode='r') as h5_file:
            stats_file_data.dist_min = torch.tensor(h5_file['dist_min'], device=self.device, dtype=self.dtype)
            stats_file_data.dist_avg = torch.tensor(h5_file['dist_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.dist_max = torch.tensor(h5_file['dist_max'], device=self.device, dtype=self.dtype)
            stats_file_data.prop_ict = torch.tensor(h5_file['prop_ict'], device=self.device, dtype=self.dtype)
            stats_file_data.w2wd_min = torch.tensor(h5_file['w2wd_min'], device=self.device, dtype=self.dtype)
            stats_file_data.w2wd_avg = torch.tensor(h5_file['w2wd_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.w2wd_max = torch.tensor(h5_file['w2wd_max'], device=self.device, dtype=self.dtype)
            stats_file_data.wavg_avg = torch.tensor(h5_file['wavg_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.intf_min = torch.tensor(h5_file['intf_min'], device=self.device, dtype=self.dtype)
            stats_file_data.intf_avg = torch.tensor(h5_file['intf_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.intf_max = torch.tensor(h5_file['intf_max'], device=self.device, dtype=self.dtype)
            stats_file_data.rndf_min = torch.tensor(h5_file['rndf_min'], device=self.device, dtype=self.dtype)
            stats_file_data.rndf_avg = torch.tensor(h5_file['rndf_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.rndf_max = torch.tensor(h5_file['rndf_max'], device=self.device, dtype=self.dtype)
            stats_file_data.latf_min = torch.tensor(h5_file['latf_min'], device=self.device, dtype=self.dtype)
            stats_file_data.latf_avg = torch.tensor(h5_file['latf_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.latf_max = torch.tensor(h5_file['latf_max'], device=self.device, dtype=self.dtype)
            stats_file_data.totf_min = torch.tensor(h5_file['totf_min'], device=self.device, dtype=self.dtype)
            stats_file_data.totf_avg = torch.tensor(h5_file['totf_avg'], device=self.device, dtype=self.dtype)
            stats_file_data.totf_max = torch.tensor(h5_file['totf_max'], device=self.device, dtype=self.dtype)
            stats_file_data.timestep = torch.tensor(h5_file['timestep'], device=self.device, dtype=self.dtype)
        stats_file_data.num_iterations = stats_file_data.timestep.shape[0]
        click.echo(f'Number of iterations: {stats_file_data.num_iterations}')
        stats_file_data.time = torch.zeros_like(stats_file_data.timestep)
        time : float = 0.0
        for i in range(stats_file_data.num_iterations):
            time += stats_file_data.timestep[i]
            stats_file_data.time[i] = time
        self.stats_files_data[stats_name] = stats_file_data

    # ---

    def load_timing_file(
            self,
            timing_name : str,
            file_path : str
            ) -> TimingFileData:
        """ Load a timing data file. """
        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                assert 'time_total' in f.keys()
                wall_time = f['time_total']
                wall_time = torch.tensor(
                    wall_time,
                    device=self.device,
                    dtype=self.dtype)
                i_max = torch.argmax(wall_time) 
                wall_time = wall_time[0:i_max + 1]
                num_ids :torch.Tensor = torch.arange(
                    0,
                    i_max+1,
                    device=self.device,
                    dtype=torch.int64)
                assert wall_time.shape == num_ids.shape
        elif file_path.endswith('.csv'):
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                num_steps = sum(1 for _ in csv_reader)
                wall_time = torch.empty(
                    (num_steps, ),
                    device=self.device,
                    dtype=self.dtype)
                num_ids = torch.empty(
                    (num_steps, ),
                    device=self.device,
                    dtype=torch.int64)
                csv_reader = csv.reader(csv_file, delimiter=',')
                for i, row in enumerate(csv_reader):
                    print(row)
                    if i >= num_steps:
                        break
                    wall_time[i] = float(row[0])
                    num_ids[i] = float(row[1])
        else:
            raise RuntimeError('Unrecognized file extension...')
        timing_file_data = TimingFileData()
        timing_file_data.num_iterations = num_ids.shape[0]
        timing_file_data.num_ids = num_ids
        timing_file_data.wall_time = wall_time
        self.timing_files_data[timing_name] = timing_file_data
        return timing_file_data
        
    # ---

    def compute_histogram(
            self,
            histogram_name : str,
            class_type : ClassType,
            data_type : DataType,
            sample_collection_data : SampleCollectionData,
            range_min : float,
            range_max : float,
            num_bins : int = 1024
            ) -> Histogram:
        """ Compute histogram. """
        click.echo('Calculating distances histogram...')
        data_points : torch.Tensor
        data_labels : torch.Tensor
        assert class_type != ClassType.cross
        if data_type == DataType.embedding:
            data_points = sample_collection_data.embeddings
        elif data_type == DataType.latent:
            data_points = sample_collection_data.latents
        else:
            raise
        num_samples = data_points.shape[0]
        data_labels = sample_collection_data.identities
        def batch_generator() -> list[tuple[int, int, int, int]]:
            batch = []
            for i in range(1 + num_samples // self.batch_size):
                for j in range(i, 1 + num_samples // self.batch_size):
                    i0 : int = i * self.batch_size
                    i1 : int = min(i0 + self.batch_size, num_samples)
                    j0 : int = j * self.batch_size
                    j1 : int = min(j0 + self.batch_size, num_samples)
                    batch.append((i0, i1, j0, j1))
            return batch
        bin_values : torch.Tensor | None = None
        bin_count : torch.Tensor | None = None
        num_binned : int = 0
        for i0, i1, j0, j1 in tqdm(batch_generator()):
            with torch.no_grad():
                d_i = data_points[i0 : i1]
                id_i = data_labels[i0 : i1]
                d_j = data_points[j0 : j1]
                id_j = data_labels[j0 : j1]
                distances : torch.Tensor
                if data_type == DataType.embedding:
                    distances = self.embedding.distance(d_i, d_j)
                    #distances = torch.cos(distances)
                elif data_type == DataType.latent:
                    distances = torch.cdist(d_i, d_j)
                else:
                    raise
                id_i = einops.repeat(id_i, 'n -> n m', m=d_j.shape[0])
                id_j = einops.repeat(id_j, 'n -> m n', m=d_i.shape[0])
                assert id_i.shape == distances.shape, 'Internal error'
                assert id_j.shape == distances.shape, 'Internal error'
                mask : torch.Tensor
                if class_type == ClassType.intra:
                    mask = torch.where(id_i == id_j, True, False)
                elif class_type == ClassType.inter:
                    mask = torch.where(id_i != id_j, True, False)
                else:
                    raise
                if i0 == j0:
                    mask = mask.triu(diagonal=1)
                distances = torch.masked_select(distances, mask)
                num_binned += distances.shape[0]
                if distances.shape[0] > 0:
                    y_hist, x_hist = torch.histogram(
                        input=distances,
                        bins=num_bins,
                        range=(range_min, range_max),
                        density=False)
                    if bin_values is None:
                        bin_values = x_hist[:-1]
                        bin_count = y_hist
                    else:
                        bin_count += y_hist
        click.echo(f'-> binned values: {num_binned}')
        if num_binned > 0:
            bin_count = bin_count.to(self.dtype)
            bin_count = bin_count / float(num_binned)
        else:
            bin_count = None
            bin_values = None
        histogram = Histogram()
        histogram.class_type = class_type
        histogram.data_type = data_type
        histogram.num_samples = num_binned
        histogram.bin_values = bin_values
        histogram.bin_count = bin_count
        self.histograms[histogram_name] = histogram
        return histogram

    # ---

    def compute_cross_histogram(
            self,
            histogram_name : str,
            data_type : DataType,
            sample_collection_data : SampleCollectionData,
            cross_collection_data : SampleCollectionData,
            range_min : float,
            range_max : float,
            num_bins : int = 1024
            ) -> Histogram:
        """ Compute histogram. """
        click.echo('Calculating distances cross histogram...')
        data_points_i : torch.Tensor
        data_points_j : torch.Tensor
        if data_type == DataType.embedding:
            data_points_i = sample_collection_data.embeddings
            data_points_j = cross_collection_data.embeddings
        elif data_type == DataType.latent:
            data_points_i = sample_collection_data.latents
            data_points_j = cross_collection_data.latents
        else:
            raise
        num_samples_i = data_points_i.shape[0]
        num_samples_j = data_points_j.shape[0]
        def batch_generator() -> list[tuple[int, int, int, int]]:
            batch = []
            for i in range(1 + num_samples_i // self.batch_size):
                for j in range(1 + num_samples_j // self.batch_size):
                    i0 : int = i * self.batch_size
                    i1 : int = min(i0 + self.batch_size, num_samples_i)
                    j0 : int = j * self.batch_size
                    j1 : int = min(j0 + self.batch_size, num_samples_j)
                    batch.append((i0, i1, j0, j1))
            return batch
        bin_values : torch.Tensor | None = None
        bin_count : torch.Tensor | None = None
        num_binned : int = num_samples_i * num_samples_j
        for i0, i1, j0, j1 in tqdm(batch_generator()):
            with torch.no_grad():
                d_i = data_points_i[i0 : i1]
                d_j = data_points_j[j0 : j1]
                distances : torch.Tensor
                if data_type == DataType.embedding:
                    distances = self.embedding.distance(d_i, d_j)
                elif data_type == DataType.latent:
                    distances = torch.cdist(d_i, d_j)
                else:
                    raise
                y_hist, x_hist = torch.histogram(
                    input=distances,
                    bins=num_bins,
                    range=(range_min, range_max),
                    density=False)
                if bin_values is None:
                    bin_values = x_hist[:-1]
                    bin_count = y_hist
                else:
                    bin_count += y_hist
        click.echo(f'-> binned values: {num_binned}')
        bin_count = bin_count.to(self.dtype)
        bin_count = bin_count / float(num_binned)
        histogram = Histogram()
        histogram.class_type = ClassType.cross
        histogram.data_type = data_type
        histogram.num_samples = num_binned
        histogram.bin_values = bin_values
        histogram.bin_count = bin_count
        self.histograms[histogram_name] = histogram

    # ---

    def plot_histograms(
            self,
            plot_name : str,
            histograms : list[Histogram]
            ) -> PlotItem:
        """ Generate a plot from a list of histograms. """
        assert plot_name is not None
        assert plot_name not in self.plot_items.keys(), 'plot name already exists'
        data_type : DataType | None = None
        value_min : float | None = None
        value_max : float | None = None
        curves : list[PlotCurve] = [] 
        for histogram in histograms:
            if data_type is None:
                data_type = histogram.data_type
            else:
                assert histogram.data_type == data_type, \
                    'Histograms plots should have the same data type'
            histogram_min = float(torch.min(histogram.bin_values))
            value_min = histogram_min if value_min is None else min(value_min, histogram_min)
            histogram_max = float(torch.max(histogram.bin_values))
            value_max = histogram_max if value_max is None else max(value_max, histogram_max)
            curve = PlotCurve()
            curve.x_values = histogram.bin_values
            curve.y_values = histogram.bin_count
            curves.append(curve)
        plot_item = PlotItem()
        plot_item.curves = curves
        plot_item.x_axis.value_min = value_min
        plot_item.x_axis.value_max = value_max
        return plot_item

    # ---

    def show(
            self
            ) -> torch.Tensor:
        """ Show the plots via pyqtgraph. """
        self.app = pg.mkQApp("plot")
        self.win = pg.GraphicsLayoutWidget(show=True, title="plot")
        self.win.resize(1200,800)
        self.win.setBackground('w')
        pg.setConfigOptions(antialias=True)
        self.plt = self.win.addPlot()
            #labels={'left': plot_item.x_axis.label, 'bottom': plot_item.y_axis.label})
        self.plt.addLegend()
        styles = \
        [
            QtCore.Qt.SolidLine, 
            QtCore.Qt.DashLine, 
            QtCore.Qt.DotLine, 
            QtCore.Qt.DashDotLine,
            QtCore.Qt.DashDotDotLine
        ]
        i_style : int = 0
        for plot_item_name in self.plot_items.keys():
            plot_item : PlotItem = self.plot_items[plot_item_name]
            for curve in plot_item.curves:
                curve : PlotCurve
                self.plt.plot(
                    curve.x_values,
                    curve.y_values,
                    pen=pg.mkPen(
                        'g',
                        width = 2,
                        style=styles[i_style]),
                    name=plot_item_name)
            i_style += 1
            if i_style == len(styles):
                i_style = 0
        self.plt.show()
        pg.exec()

# ---

@click.group(
    help='Plot data',
    chain=True)
@click.pass_context
@click.option(
    '--embedding-type',
    '-et',
    help='Embedding type',
    type=click.Choice(Embedding.get_available_models()),
    default='iresnet50',
    show_default=True)
@click.option(
    '--batch-size',
    '-b',
    default=4096,
    type=int,
    help='Processing batch size',
    show_default=True)
def plot(
        ctx : click.Context,
        embedding_type : list[str],
        batch_size : int
        ) -> None:
    click.echo('Initializing plot module...')
    plot = Plot(
        embedding_type=embedding_type,
        batch_size=batch_size)
    ctx.obj = plot
    click.echo('... done')

# ---

@plot.command(
    help='Loads a sample collection')
@click.pass_context
@click.option(
    '--collection-name',
    '-c',
    help='Name for the sample sample collection',
    type=str,
    default='collection',
    show_default=True)
@click.option(
    '--file-path',
    '-f',
    help='Path of the sample collection',
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True)
def load_sample_collection(
        ctx : click.Context,
        collection_name : str,
        file_path : str
        ) -> None:
    plot : Plot = ctx.obj
    plot.load_sample_collection(
        collection_name=collection_name,
        file_path=file_path)
    
# ---

@plot.command(
    help='Load a temporal stats file')
@click.pass_context
@click.option(
    '--stats-name',
    '-s',
    help='Name for the stats data',
    type=str,
    default='stats',
    show_default=True)
@click.option(
    '--file-path',
    '-f',
    help='Path to the data file',
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True)
def load_stats_file(
        ctx : click.Context,
        stats_name : str,
        file_path : str
        ) -> None:
    """ Add a stats file (CLI command) """
    plot : Plot = ctx.obj
    plot.load_stats_file(
        stats_name=stats_name,
        file_path=file_path)
    
# ---

@plot.command(
    help='Load a timing file')
@click.pass_context
@click.option(
    '--timing-name',
    '-t',
    help='Name for the timing data',
    type=str,
    default='timing',
    show_default=True)
@click.option(
    '--file-path',
    '-f',
    help='Path to the data file',
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True)
def load_timing_file(
        ctx : click.Context,
        data_name : str,
        file_path : str
        ) -> None:
    plot : Plot = ctx.obj
    plot.load_timing_file(
        data_name=data_name,
        file_path=file_path)

# ---

@plot.command(
    help='Compute an embedding or latent histogram')
@click.pass_context
@click.option(
    '--histogram-name',
    '-h',
    help='Name for the histogram',
    type=str,
    default='histogram',
    show_default=True)
@click.option(
    '--data-type',
    '-dt',
    help='Data type',
    type=click.Choice([data_type.name for data_type in DataType]),
    required=True)
@click.option(
    '--class-type',
    '-ct',
    help='Class type',
    type=click.Choice([class_type.name for class_type in ClassType]),
    required=True)
@click.option(
    '--collection-name',
    '-c',
    help='Name of the source sample collection',
    type=str,
    default='collection',
    show_default=True)
@click.option(
    '--cross-collection-name',
    '-x',
    help='Name of the secondary sample collection (for cross histograms)',
    type=str,
    default=None,
    show_default=True)
@click.option(
    '--range-min',
    '-min',
    help='Minimum value of the data range',
    type=float,
    default=0.0,
    show_default=True)
@click.option(
    '--range-max',
    '-max',
    help='Maximum value of the data range',
    type=float,
    default=math.pi,
    show_default=True)
@click.option(
    '--export',
    '-e',
    help='Export data to text file',
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
    default=None)
def compute_histogram(
        ctx : click.Context,
        histogram_name : str,
        data_type : str,
        class_type : str,
        collection_name : str,
        cross_collection_name : str | None,
        range_min : float,
        range_max : float,
        export : str | None
        ) -> None:
    plot : Plot = ctx.obj
    data_type = DataType[data_type]
    class_type = ClassType[class_type]
    sample_collection_data = plot.sample_collections_data[collection_name]
    histogram : Histogram
    if class_type == ClassType.cross:
        assert cross_collection_name is not None
        cross_collection_data = plot.sample_collections_data[cross_collection_name]
        histogram = plot.compute_cross_histogram(
            histogram_name=histogram_name,
            data_type=data_type,
            sample_collection_data=sample_collection_data,
            cross_collection_data=cross_collection_data,
            range_min=range_min,
            range_max=range_max)
    else:
        cross_collection_data = None
        histogram = plot.compute_histogram(
            histogram_name=histogram_name,
            data_type=data_type,
            class_type=class_type,
            sample_collection_data=sample_collection_data,
            range_min=range_min,
            range_max=range_max)
    if export is not None:
        dat_file = DatFile()
        dat_file.add_column(histogram.bin_values)
        dat_file.add_column(histogram.bin_count)
        dat_file.export(export)

# ---

@plot.command(
    help='Compute an embedding or latent histogram')
@click.pass_context
@click.option(
    '--plot-name',
    '-p',
    help='Name of the plot',
    type=str,
    default='plot',
    show_default=True)
@click.option(
    '--add-histogram',
    '-a',
    help='Name(s) of the histogram to plot (multiple values possible)',
    type=str,
    default=('histogram', ),
    multiple=True)
def plot_histograms(
        ctx : click.Context,
        plot_name : str,
        add_histogram : tuple[str]
        ) -> None:
    plot : Plot = ctx.obj
    histograms = []
    for histogram_name in add_histogram:
        assert histogram_name in plot.histograms.keys(), 'Histogram name does not exist'
        histograms.append(plot.histograms[histogram_name])
    plot.plot_histograms(
        plot_name=plot_name,
        histograms=histograms)

# ---

@plot.command(
    help='Show the plots via pyqtgraph')
@click.pass_context
def show(
        ctx : click.Context,
        ) -> None:
    plot : Plot = ctx.obj
    plot.show()
