#!/bin/bash
#$ -N create_env
#$ -l mem=2G
#$ -cwd
#$ -pe default 2
#$ -o $HOME
#$ -e $HOME

wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda"
rm Miniconda3-latest-Linux-x86_64.sh

export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
eval "$(conda shell.bash hook)"
conda create -y -q --name iannwtf_final python=3.9

conda activate MLinPractice
conda install -y -q -c conda-forge matplotlib=3.4.3
conda install -y -q -c conda-forge pandas=1.1.5
conda install -y -q -c conda-forge numpy=1.19.5
conda install -y -q -c conda-forge mlflow=1.20.2
conda install -y -q -c conda-forge tqdm=4.62.3
conda install -y -q -c conda-forge tensorflow=2.6.0
conda install -y -q -c conda-forge tensorflow-datasets=4.1.0
conda install -y -q -c conda-forge pretty-midi=0.2.9
conda install -y -q -c conda-forge scipy=1.7.1
conda install -y -q -c conda-forge pyFluidSynth=1.3.0
conda deactivate

cd $HOME/miniconda/pkgs
rm *.tar.bz2 -f 2> /dev/null