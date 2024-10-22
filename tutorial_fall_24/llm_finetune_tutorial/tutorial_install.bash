#!/bin/bash

# ---- Installing Jupyter Kernel ----

# path to folder with jupyter notebook kernel configuration
SRC_DIR=/home1/10156/gj3385/llm_finetune_tutorial/config_files/jupyter_kernel_config

# path where local jupyter kernels are installed to
DST_DIR=~/.local/share/jupyter/kernels/

# make install directory
mkdir -p $DST_DIR

# copy our configuration to install directory
cp -r $SRC_DIR $DST_DIR


# ---- Copying Notebook and Materials to $HOME ----

# path to folder that holds the notebook and other materials
SRC_DIR=/home1/10156/gj3385/llm_finetune_tutorial

# copy materials to local HOME directory
cp -r $SRC_DIR $HOME/


