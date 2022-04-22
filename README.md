# DFCI CODEX Pipeline

**Molecular Imaging Core Facility**  
**Dana-Farber Cancer Institute**

This repository contains the customized analysis pipeline developed to support CODEX imaging at Dana-Farber
It is built using the [`PathML`](https://github.com/Dana-Farber-AIOS/pathml) toolkit for computational pathology.

Prior to using this pipeline, images should be preprocessed using the CODEX Processor software v1.8.1.9 (Akoya Biosciences), which will 
be responsible for background subtraction, deconvolution, shading correction, etc. These steps are **NOT** included in the pipeline in this repo.

This pipeline then performs cell segmentation, quantifies the marker levels for each cell, and produces a counts matrix for downstream analysis.

## Instructions

### 0. Environment setup

Before running, the environment must be set up with all required dependencies.
Conda is the recommended tool for environment management. 
Download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)

Create conda environment with required dependencies and install pathml:
````
conda env create -f environment.yml
conda activate codex
pip install pathml
````
 
Please refer to the [PathML documentation](https://github.com/Dana-Farber-AIOS/pathml) for complete instructions.  


### 1. Stack images into a single image

The expected input to the pipeline is a single image containing all channels, in a format such as `.tif`, `.tiff`, 
`.ome.tif`, `.qptiff`, or one of the 150+ other file formats supported by PathML 
(documentation here)[https://pathml.readthedocs.io/en/latest/loading_slides.html#supported-file-formats]. 

If the images for each channel are saved in separate files, use the `bfconvert` tool to 
first stack them into a single image, before running pipeline:

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
    --tile-size 2048 \
    --prefix testing
````

Running `python codex-pathml.py --help` will display more information about how to use the command line tool.

The pipeline will then run, saving the resulting counts matrix in `.csv` for use with the MAV viewer.
The saved `.h5path` file contains a counts matrix in `AnnData` format to streamline downstream analysis in 
any of the tools from the single-cell analysis ecosystem, such as ScanPy, Seurat, etc.
