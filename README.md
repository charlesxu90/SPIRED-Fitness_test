# SPIRED-Fitness_test

## Prepare environment
Following [SPIRED-Fitness](https://github.com/Gonglab-THU/SPIRED-Fitness).


```shell
# Install SPIRED-Fitness
mamba create -n spired_fitness python=3.11
conda activate spired_fitness
pip install click einops pandas biopython tqdm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Download the [model parameters](https://zenodo.org/doi/10.5281/zenodo.10589085) and move it into the `data/model` folder

## Run
```shell
conda activate spired_fitness
python run_SPIRED-Stab.py --fasta_file data/example/test.fasta --saved_folder results/example
```