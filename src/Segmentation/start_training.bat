@ECHO off

:: EDIT these paths to your directories, respectively:

:: MaskRCNN folder (delete the word "rem" from the beginning of the line and set the path if you already have
:: a MaskRCNN folder - in this case, move the word "rem" from between the beginning of the last 2 lines):

set "pathToInsert=C:\\Users\\jaspe\\Downloads\\biomagdsb-master\\biomagdsb-master\\Mask_RCNN"

:: working directory where you downloaded the code and will have the output under ~\kaggle_workflow\outputs\maskrcnn:
set root_dir=C:\\Users\\jaspe\\Downloads\\biomagdsb-master\\biomagdsb-master

:: directory of your images to segment:
set images_dir=C:\\Users\\jaspe\\Downloads\\biomagdsb-master\\biomagdsb-master\\badSegment_1_SUBIMAGES

:: directory of your python 3.6 virtual environment:
set pyVirtPath="C:\\Users\\jaspe\\anaconda3\\envs\\tensorold"
:: -----------------------------------------------------------------------------


:: --- DO NOT EDIT from here ---
run_workflow_trainOnly.bat %root_dir% %images_dir% %pyVirtPath% %pathToInsert%
run_workflow_trainOnly.bat %root_dir% %images_dir% %pyVirtPath%