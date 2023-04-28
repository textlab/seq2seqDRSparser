# Sequence-to-sequence transformers-based DRS parser experiments

This folder has the code to reproduce the results of the paper accepted to IWCS 2023, "Experiments in training transformer sequence-to-sequence DRS parsers".

To run, conda installation is required. Conda activate and deactivate commands are needed. If you don't have conda, please intall anaconda from  https://www.anaconda.com/ .

    conda create -n drs python=3.6

    conda activate drs

    pip3 install -r requirements.txt

    cd other_repos

    ./setup.sh

Then in the main directory:

    conda activate drs

    ./download_necessary_files.sh

    python3 process_data.py

    python3 train_tokenizers.py 

    python3 train_models.py

This work is licensed under [Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
