@ECHO off

:: EDIT these paths to your directories, respectively:

:: MaskRCNN folder (delete the word "rem" from the beginning of the line and set the path if you already have
:: a MaskRCNN folder - in this case, move the word "rem" from between the beginning of the last 2 lines):
set "pathToInsert=C:\\Users\\jaspe\\Documents\\Bioinformatics\\ML\\biomagdsb-master\\biomagdsb-master\\Mask_RCNN"

:: working directory where you downloaded the code and will have the output under ~\kaggle_workflow\outputs\maskrcnn:
set root_dir=C:\\Users\\jaspe\\Documents\\Bioinformatics\\ML\\biomagdsb-master\\biomagdsb-master

:: directory of your images to segment:
set images_dir=C:\\Users\\jaspe\\Documents\\Bioinformatics\\ML\\biomagdsb-master\\biomagdsb-master\\largeTest
:: -----------------------------------------------------------------------------


:: --- DO NOT EDIT from here ---
run_workflow_predictOnly_full.bat %root_dir% %images_dir% %pathToInsert%
run_workflow_predictOnly_full.bat %root_dir% %images_dir%129*+-+-+-+-++++++++++++