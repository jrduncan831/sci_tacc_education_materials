#!/bin/bash

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
