#!/usr/bin/bash
matlab -nodisplay -nodesktop -r "data_processing.m"

cd /Users/xvz5220-admin/Documents/cell_tracking_other_script/conservation_tracking_pipeline/scripts/
INPUT_FOLDER=/Users/xvz5220-admin/Dropbox/cell_tracking_data/data_output/01/

python convertTiffToRGBPng.py $INPUT_FOLDER $INPUT_FOLDER 4096




# Convert the C2DL FOLDER
cd /Users/xvz5220-admin/Documents/cell_tracking_other_script/conservation_tracking_pipeline/scripts/

INPUT_FOLDER=/Users/xvz5220-admin/Dropbox/cell_tracking_data/C2DL/02_GT/SEG/
python convertTiffToRGBPng.py $INPUT_FOLDER $INPUT_FOLDER 4096

