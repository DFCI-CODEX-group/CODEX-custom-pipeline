# Jacob Rosenthal
# script for codex workflow based on PathML

import argparse
import pathlib
from pathlib import Path
import time
from datetime import timedelta

from pathml.core import CODEXSlide
from pathml.preprocessing import Pipeline, CollapseRunsCODEX, SegmentMIF, QuantifyMIF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CODEX analysis pipeline')
    parser.add_argument('--inputfile', required = True, type = pathlib.Path, help = 'path to input tiff file')
    parser.add_argument('--nucleus_marker_cycle_index', required = True, type = int, dest = "nuc_cyc_ix",
                        help = 'cycle index for nucleus marker (0 indexed)')
    parser.add_argument('--nucleus_marker_channel_index', required = True, type = int, dest = "nuc_chan_ix",
                        help = 'channel index for nucleus marker (0 indexed)')
    parser.add_argument('--cytoplasm_marker_cycle_index', required = True, type = int, dest = "cyto_cyc_ix",
                        help = 'cycle index for cytoplasm marker (0 indexed)')
    parser.add_argument('--cytoplasm_marker_channel_index', required = True, type = int, dest = "cyto_chan_ix",
                        help = 'channel index for cytoplasm marker (0 indexed)')

    args = parser.parse_args()

    # load slide and extract as region
    path = Path(args.inputfile)

    slide = CODEXSlide(str(path))
    print(f"loaded image of shape {slide.shape}")

    # convert cycle and channel indices into indices for collapsed array
    n_channels = slide.slide.shape_list[0][2]
    n_cycles = slide.slide.shape_list[0][4]

    nucleus_marker_index = args.nuc_cyc_ix * n_cycles + args.nuc_chan_ix
    cytoplasm_marker_index = args.cyto_cyc_ix * n_cycles + args.cyto_chan_ix

    # Define a pipeline
    pipe = Pipeline([
        CollapseRunsCODEX(z = 0),
        SegmentMIF(model='mesmer',
                   nuclear_channel=nucleus_marker_index,
                   cytoplasm_channel=cytoplasm_marker_index,
                   image_resolution=0.5),
        QuantifyMIF(segmentation_mask='cell_segmentation')
    ])

    print("Starting pipeline...")
    t1 = time.time()

    slide.run(pipe, distributed = False, tile_size= slide.shape, tile_pad=False)

    t2 = time.time()
    print(f"Finished running pipeline ({str(timedelta(seconds = t2 - t1))})")

    print(f"Counts matrix generated: {slide.counts.shape[0]} cells, {slide.counts.shape[1]} markers")

    print(f"Saving counts matrix to: {path.name}.h5ad")
    # write the count matrix to file
    slide.counts.write(f"{path.name}.h5ad")

    del slide
    print("done")
