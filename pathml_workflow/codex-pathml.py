import argparse
import pathlib
from pathlib import Path
import time
from datetime import timedelta
import json
import os
import numpy as np
import pandas as pd
import csv
import sys
import skimage
import javabridge
# silence copious tensorflow warnings
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import pathml
from pathml.core import CODEXSlide
from pathml.preprocessing import Pipeline, CollapseRunsCODEX, SegmentMIF, QuantifyMIF


class FilterEdgeCells(pathml.preprocessing.transforms.Transform):
    """
    Filter out cells from counts matrix which are within certain number of px from edge of tile.
    When using overlapping tiles, this can be used to avoid double-counting cells that occur in the overlap.

    Args:
        edge_distance (int): distance from edge at which cells are to be filtered (e.g. 100px)
        slide_shape (tuple): dimensions of wsi (e.g. slide.shape).
            Used to check whether an individual tile is on the edge or not.
    """
    def __init__(self, edge_distance, slide_shape):
        self.edge_distance = edge_distance
        self.slide_shape = slide_shape

    def apply(self, tile):
        # first check if tile is on the edge
        i, j = tile.coords
        di, dj = tile.shape[0:2]
        edge_top = i == 0
        edge_bottom = i + di > self.slide_shape[0]
        edge_left = j == 0
        edge_right = j + dj > self.slide_shape[1]

        # this logic filters out obs around each edge
        # the or condition in each row keeps edge obs, if it's an edge tile
        counts = tile.counts.copy()
        newcounts = counts[((counts.obs.y > i + self.edge_distance) | edge_top) &
                           ((counts.obs.y < i + tile.shape[0] - self.edge_distance) | edge_bottom) &
                           ((counts.obs.x > j + self.edge_distance) | edge_left) &
                           ((counts.obs.x < j + tile.shape[1] - self.edge_distance) | edge_right)]
        tile.counts = newcounts.copy()


class MembraneMarkerWatershed(pathml.preprocessing.transforms.Transform):
    """
    Performs marker-controlled watershed segmentation.
    Uses nuclei segmentation as markers. Adds result to tile as "watershed" segmentation mask.
    Wraps ``skimage.segmentation.watershed``

    Args:
        membrane_channel (int): index of marker to use for filling with watershed algorithm
        marker_segmentation_mask (str): name of segmentation mask to use as markers
        watershed_line (bool): If watershed_line is True, a one-pixel wide line separates the regions obtained
          by the watershed algorithm. The line has the label 0. Defaults to True.
        mask_name (str): name for new mask. Defaults to "watershed"
    """
    def __init__(self, membrane_channel, marker_segmentation_mask, watershed_line=True, mask_name="watershed"):
        self.membrane_channel = membrane_channel
        self.marker_segmentation_mask = marker_segmentation_mask
        self.watershed_line = watershed_line
        self.mask_name = mask_name

    def apply(self, tile):
        watershed = skimage.segmentation.watershed(
            image = tile.image[..., self.membrane_channel],
            markers = tile.masks[self.marker_segmentation_mask].squeeze(2),
            watershed_line = self.watershed_line
        )
        tile.masks[self.mask_name] = watershed[..., np.newaxis]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CODEX analysis pipeline')
    parser.add_argument('--input-image', required = True, type = pathlib.Path, dest = "inputfile",
                        help = 'path to input stacked tiff file')
    parser.add_argument('--metadata', required = True, type = pathlib.Path,
                        help = 'path to experiment metadata (`experiment.json`)')
    parser.add_argument('--nucleus_marker_cycle_index', required = True, type = int, dest = "nuc_cyc_ix",
                        help = 'cycle index for nucleus marker (0 indexed)')
    parser.add_argument('--nucleus_marker_channel_index', required = True, type = int, dest = "nuc_chan_ix",
                        help = 'channel index for nucleus marker (0 indexed)')
    parser.add_argument('--cytoplasm_marker_cycle_index', required = True, type = int, dest = "cyto_cyc_ix",
                        help = 'cycle index for cytoplasm marker (0 indexed)')
    parser.add_argument('--cytoplasm_marker_channel_index', required = True, type = int, dest = "cyto_chan_ix",
                        help = 'channel index for cytoplasm marker (0 indexed)')
    parser.add_argument('--tile-size', required = False, default = 1024, type = int, dest = "tile_size",
                        help = 'tile size (pixels)')
    parser.add_argument('--tile-overlap', required = False, default = 200, type = int, dest = "tile_overlap",
                        help = 'overlap between tiles (pixels)')
    args = parser.parse_args()

    print(f"working dir: {os.getcwd()}")
    print(f"pathml Version: {pathml.__version__}")
    # load metadata, get channel names
    metadata_p = Path(args.metadata)
    if not (metadata_p.is_file() and metadata_p.suffix == ".json"):
        raise ValueError(f"Input metadata file invalid: {args.metadata}")
    try:
        with open(metadata_p) as f:
            experiment_metadata = json.load(f)
            channel_names = experiment_metadata["channelNames"]["channelNamesArray"]
            experiment_name = experiment_metadata["name"]
    except:
        raise Exception(f"Failed loading channel names and experiment name from metadata file: {args.metadata}")

    # need to uniquify channel names, e.g. "Blank", "Blank" --> "Blank-1", "Blank-2"
    # from: https://stackoverflow.com/a/30650847/17836677
    newlist = []
    for i, v in enumerate(channel_names):
        totalcount = channel_names.count(v)
        count = channel_names[:i].count(v)
        newlist.append(v + "-" + str(count + 1) if totalcount > 1 else v)
    channel_names = newlist

    # load slide
    path = Path(args.inputfile)
    slide = CODEXSlide(str(path))
    print(f"loaded image of shape {slide.slide.shape_list[0]}")

    n_channels = slide.slide.shape_list[0][3]
    n_cycles = slide.slide.shape_list[0][4]
    print(f"n channels: {n_channels}\tn cycles: {n_cycles}")

    # for array where channels are rows, and cycles are columns:
    # PathML collapses CODEX runs using row-major indexing, while CODEX processor uses column-major
    # Need to be careful when indexing channels, and convert when needed
    channel_map = []
    for chan_ix in range(n_channels):
        for cyc_ix in range(n_cycles):
            pathml_ix = cyc_ix * n_channels + chan_ix
            channel_map.append(pathml_ix)

    # channel names ordered as they are in PathML after CollapseRunsCODEX
    channel_names_pathml = [channel_names[i] for i in channel_map]

    nucleus_marker_index = args.nuc_chan_ix * n_cycles + args.nuc_cyc_ix
    cytoplasm_marker_index = args.cyto_chan_ix * n_cycles + args.cyto_cyc_ix

    print(f"Using nucleus marker: {channel_names_pathml[nucleus_marker_index]}\tPathml index: {nucleus_marker_index}")
    print(f"Using cytoplasm marker: {channel_names_pathml[cytoplasm_marker_index]}\tPathml index: {cytoplasm_marker_index}")

    # Define the pipeline
    pipe = Pipeline([
        CollapseRunsCODEX(z = 0),
        SegmentMIF(
            model = 'mesmer',
            nuclear_channel = nucleus_marker_index,
            cytoplasm_channel = cytoplasm_marker_index,
            image_resolution = 0.5),
        MembraneMarkerWatershed(
            membrane_channel = cytoplasm_marker_index,
            marker_segmentation_mask = "nuclear_segmentation",
            watershed_line = True,
            mask_name = "watershed"),
        QuantifyMIF(segmentation_mask = 'watershed'),
        FilterEdgeCells(edge_distance = args.tile_overlap / 2, slide_shape = slide.shape)
    ])

    print(f"Starting pipeline with tile size {args.tile_size}, tile overlap {args.tile_overlap}...")
    t1 = time.time()

    slide.run(pipe,
              distributed = False,
              tile_size = args.tile_size,
              tile_stride = args.tile_size - args.tile_overlap,
              tile_pad = True,
              normalize = False)

    t2 = time.time()
    print(f"Finished running pipeline ({str(timedelta(seconds = t2 - t1))})")

    print(f"Counts matrix generated: {slide.counts.shape[0]} cells, {slide.counts.shape[1]} markers")

    # convert counts matrix to CSV in format for MAV. Formatting for MAV is very particular
    counts = slide.counts.to_memory()
    mav_df = pd.DataFrame(counts.X, columns = [name + " Nucleus Intensity" for name in channel_names_pathml])
    # reorder back to CODEX channel order
    mav_df = mav_df.iloc[:, [channel_map.index(i) for i in range(len(channel_map))]]
    mav_df["XMin"] = counts.obs.x.values
    mav_df["XMax"] = counts.obs.x.values
    mav_df["YMin"] = counts.obs.y.values
    mav_df["YMax"] = counts.obs.y.values
    mav_df['Cell ID'] = mav_df.index
    mav_df['Object ID'] = mav_df.index
    fname = f"reg001_{experiment_name}.csv"
    mav_df.to_csv(fname, quoting = csv.QUOTE_ALL)
    print(f"Saved counts matrix to: {fname}")

    slide.write(f"{experiment_name}.h5path")
    print(f"Saved h5path to: {experiment_name}.h5path")

    javabridge.kill_vm()
