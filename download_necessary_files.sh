#!/bin/sh
wget https://raw.githubusercontent.com/utkd/encdecmodel-hf/master/data.py
cd ./data
mkdir source
cd source
wget https://pmb.let.rug.nl/releases/exp_data_2.2.0.zip
wget https://pmb.let.rug.nl/releases/exp_data_3.0.0.zip
wget https://pmb.let.rug.nl/releases/exp_data_4.0.0.zip
unzip exp_data_2.2.0.zip
unzip exp_data_3.0.0.zip
unzip exp_data_4.0.0.zip
mv pmb_exp_data_3.0.0 exp_data_3.0.0
