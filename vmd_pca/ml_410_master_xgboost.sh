#!/bin/sh

ulimit -s unlimited
. /opt/intel/parallel_studio_xe_2020/psxevars.sh intel64
export LD_LIBRARY_PATH=/opt/netcdf413_intel/lib:$LD_LIBRARY_PATH
eval "$(/csg/shzhou/anaconda3/bin/conda shell.bash hook)"
conda activate pytorch
python /ws_correct_ML/vmd_pca/ml_410_master_xgboost.py 
