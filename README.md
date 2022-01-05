# DFCI CODEX Pipeline

**Molecular Imaging Core Facility**  
**Dana-Farber Cancer Institute**

This repository contains the customized analysis pipeline developed to support CODEX imaging at Dana-Farber

Prior to using this pipeline, images should be preprocessed using the CODEX Processor software v1.8.1.9 (Akoya Biosciences), which will 
be responsible for background subtraction, deconvolution, shading correction, etc. These steps are **NOT** included in the pipeline in this repo.

This pipeline then performs cell segmentation, quantifies the marker levels for each cell, and produces a counts matrix for downstream analysis.

## Instructions

#### 0. Environment setup

- Currently code is Windows-specific (relies on a set of .bat files to run workflows)  
- Install packages into project environment: `pip install -r Segmentation/requirements.txt`

#### 1. Download code from [Dropbox](https://www.dropbox.com/sh/mrx4tzks817z36w/AAAg_c6DY5DpkIDvEYxvPJawa?dl=0)  
 
- This is the code for the current state of the pipeline
- Currently has some files that are too big to upload to GitHub directly

#### 2. Download video tutorial from [Dropbox](https://www.dropbox.com/s/4teqappe78cde2t/segmentation_final.mp4?dl=0)

- In the video, Jasper gives step-by-step instructions for running the pipeline

* need to give credit to segmentation devs: https://github.com/spreka/biomagdsb
