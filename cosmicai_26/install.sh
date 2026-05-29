#!/bin/bash

module purge

# Load specific modules to match the exact target environment
module load ucc/1.7.0 ucx/1.20.0 cmake/4.1.1 TACC gcc/15.1.0 openmpi/5.0.5 python3/3.11.8 sqlite/3.46.1 cuda/13.1

# ---- Installing Jupyter Kernel ----

# path to folder with jupyter notebook kernel configuration
SRC_DIR=/scratch/07980/sli4/containers/share/cosmicai26

# path where local jupyter kernels are installed to
DST_DIR=~/.local/share/jupyter/kernels/

# newly create directory if copy is successful
NEW_DIR=${DST_DIR}jupyter_kernel_config


if [ -e $NEW_DIR ]
  then
   echo "jupyter notebook kernel configuration already copied"
  else
   # make install directory
   mkdir -p $DST_DIR
   # copy materials to local tutorial directory
   cp -r $SRC_DIR $DST_DIR
fi

# ---- Copying Notebook and Materials to $HOME ----

# NOTE: to be updated once we have a

#path to folder that holds the notebook and other materials
SRC_DIR=/scratch/projects/tacc/sci-training/sci_tacc_education_materials/cosmicai_26

# path where tutorial content saved to
DST_DIR=$HOME

# newly create directory if copy is successful
NEW_DIR=${DST_DIR}/tutorial_spring_2026


if [ -e $NEW_DIR ]
   then
    echo "course material already copied"
   else
    # make tutorial directory
    mkdir -p $DST_DIR
    # copy materials to local tutorial directory
    cp -r $SRC_DIR $DST_DIR
fi

# reset users default environment
#module reset
#module load gcc python3
#module save
#echo "default environment changed to include gcc and python3"