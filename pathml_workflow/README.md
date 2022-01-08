# Prototype of Streamlined CODEX workflow for Molecular Imaging Core

## Overview

This is a streamlined CODEX pipeline which can be run with only 2 steps.  
It is built using the [`PathML`](https://github.com/Dana-Farber-AIOS/pathml) toolkit.

Before running, the environment must be set up with all required dependencies. 
Please Refer to the PathML documentation for complete instructions.  


## Instructions

### 1. Crop and stack tiles into a single image
 - Download bftools from Conda: `conda install -c ome bftools`
 - Create a pattern file: `echo "tiles/reg001_X03_Y02/reg001_X03_Y02_t<001-005>_z001_c<001-004>.tif" > crop.pattern`
 - Run conversion and crop with bfconvert: `bfconvert -crop 0,0,512,512 crop.pattern "cropped/test_crop512.tif"`
    
### 2. Run PathML CODEX pipeline.

Execute the pipeline with a single command. Make sure to pass as arguments the name of the input file (generated in 
step 1) as well as the cycle and channel indices for the nuclear marker (e.g. DAPI) and the cytoplasm marker 
(e.g. MHC-I)

Example usage:

````
 python codex-pathml.py \
   --inputfile /Users/jacobrosenthal/data/molecular_imaging_core/processed_2021-11-15/cropped/test.ome.tif \
   --nucleus_marker_cycle_index 0 \
   --nucleus_marker_channel_index 0 \
   --cytoplasm_marker_cycle_index 3 \
   --cytoplasm_marker_channel_index 3
````

The pipeline will then run, saving the resulting counts matrix in `.h5ad` format for downstream analysis in 
any of the tools from the single-cell analysis ecosystem, such as ScanPy, Seurat, etc.
