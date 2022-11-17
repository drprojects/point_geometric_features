#!/bin/bash

# Local variables
PROJECT_NAME=pgeof
PYTHON=3.8


# Installation script for Anaconda3 environments
echo "#############################################"
echo "#                                           #" 
echo "#           Deep View Aggregation           #"
echo "#                 Installer                 #"
echo "#                                           #" 
echo "#############################################"
echo
echo


echo "_______________ Prerequisites _______________"
echo "  - conda"
echo "  - cmake"
echo
echo


echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath ~/anaconda3`

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo


echo "________________ Installation _______________"
echo

# Create deep_view_aggregation environment from yml
conda create --name $PROJECT_NAME python=$PYTHON -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}

# Dependencies
conda install pip numpy -y
conda install -c anaconda boost -y
conda install -c omnia eigen3 -y
conda install eigen -y
conda install -c r libiconv -y

ln -s $CONDA_PREFIX/lib/python$PYTHON/site-packages/numpy/core/include/numpy $CONDA_PREFIX/include/numpy
cd src
cmake . -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython$PYTHON.so -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python$PYTHON -DBOOST_INCLUDEDIR=$CONDA_PREFIX/include -DEIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3
make
cd ..
