# Prototype of Streamlined CODEX workflow for Molecular Imaging Core

## Overview

This is a streamlined CODEX pipeline which can be run with only 2 steps.  
It is built using the [`PathML`](https://github.com/Dana-Farber-AIOS/pathml) toolkit.

Before running, the environment must be set up with all required dependencies. 
Please Refer to the PathML documentation for complete instructions.  


## Instructions

### 1. Stack images into a single image

The CODEX processor outputs each channel as a separate image. Before running this pipeline, use `bfconvert` tool to 
stack them into a single image.

- Download bftools from Conda: `conda install -c ome bftools`
- rename the files in the `stitched/` folder to follow standard pattern:
  - no marker names in filenames (manually delete for now - we can automate this later)
  - Substitute "cyc" for "t": `rename 's/^cyc/t/' cyc*`. 
    This is important so that BioFormats recognizes that cycles should be associated with time.
  - Images should now be named: "t<###>_ch<###>.tif", where ### is 001, 002, 003, etc.
- Create a pattern file: `echo "stitched/t<001-005>_ch<001-004>.tif" > stitched.pattern`
- Run conversion and crop with bfconvert: `bfconvert stitched.pattern "converted.ome.tif"`

This will stack all the channels into a single image, named `converted.ome.tif`
    
### 2. Run PathML CODEX pipeline.

Now we are ready to execute the pipeline with a single command. Make sure to pass as arguments the name of the input 
file (generated in step 1) as well as the cycle and channel indices for the nuclear marker (e.g. DAPI) 
and the cytoplasm marker (e.g. MHC-I). Note that channel indices are zero-indexed, meaning that they are counted 
as {0, 1, 2, 3, etc.}. 

Example usage:

````
python codex-pathml.py \
    --input-image Exp.\ 273\ -\ 111521SM-HNC2/processed_2021-11-15/converted/converted.ome.tiff \
    --nucleus_marker_cycle_index 0 \
    --nucleus_marker_channel_index 0 \
    --cytoplasm_marker_cycle_index 3 \
    --cytoplasm_marker_channel_index 3 \
    --metadata Exp.\ 273\ -\ 111521SM-HNC2/experiment.json \
    --tile-size 2048
````

Running `python codex-pathml.py --help` will display more information about how to use the command line tool.

The pipeline will then run, saving the resulting counts matrix in `.csv` for use with the MAV viewer.
It will also optionally save the counts matrix to `.h5ad` format to streamline downstream analysis in 
any of the tools from the single-cell analysis ecosystem, such as ScanPy, Seurat, etc.
