#!/bin/bash
condapath=$(conda info | grep -i 'base environment'|cut -d ":" -f 2 | cut -d " " -f 2)
source $condapath/etc/profile.d/conda.sh
conda activate ndrs220
cur_dir=$(pwd)
export PYTHONPATH=${cur_dir}/other_repos/Neural_DRS_220/DRS_parsing/:${PYTHONPATH}
export PYTHONPATH=${cur_dir}/other_repos/Neural_DRS_220/DRS_parsing/evaluation/:${PYTHONPATH}
"$@"
conda deactivate
