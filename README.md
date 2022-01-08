# DFCI CODEX Pipeline

**Molecular Imaging Core Facility**  
**Dana-Farber Cancer Institute**

This repository contains the customized analysis pipeline developed to support CODEX imaging at Dana-Farber

Prior to using this pipeline, images should be preprocessed using the CODEX Processor software v1.8.1.9 (Akoya Biosciences), which will 
be responsible for background subtraction, deconvolution, shading correction, etc. These steps are **NOT** included in the pipeline in this repo.

This pipeline then performs cell segmentation, quantifies the marker levels for each cell, and produces a counts matrix for downstream analysis.

## Instructions

### Download video tutorial from [Dropbox](https://www.dropbox.com/s/4teqappe78cde2t/segmentation_final.mp4?dl=0)

- In the video, Jasper gives step-by-step instructions for running the pipeline

TODO: need to give credit to segmentation devs: https://github.com/spreka/biomagdsb
