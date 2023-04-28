#!/bin/bash
condapath=$(conda info | grep -i 'base environment'|cut -d ":" -f 2 | cut -d " " -f 2)
source $condapath/etc/profile.d/conda.sh
git clone https://github.com/RikVN/Neural_DRS.git
mv Neural_DRS Neural_DRS_220
conda create -n ndrs220 python=3.6
conda activate ndrs220
cd Neural_DRS_220
git checkout v2.2.0-final
chmod +x ./src/setup.sh
./src/setup.sh
cd ..
cat 220.patch >>Neural_DRS_220/src/preprocess.py
conda deactivate
git clone https://github.com/RikVN/Neural_DRS.git
mv Neural_DRS Neural_DRS_300
conda create -n ndrs300 python=3.6
conda activate ndrs300
cd Neural_DRS_300
chmod +x ./src/setup.sh
./src/setup.sh
cd ..
conda deactivate
git clone https://github.com/RikVN/Neural_DRS.git
mv Neural_DRS Neural_DRS_400
conda create -n ndrs400 python=3.6
conda activate ndrs400
cd Neural_DRS_400
grep -v "checkout" ./src/setup.sh>temp
mv temp ./src/setup.sh
chmod +x ./src/setup.sh
./src/setup.sh
cd ..
conda deactivate
