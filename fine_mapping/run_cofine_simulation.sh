#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)


fm_input_summary_file=${1}  # real LD and peak structure of the simulated genes
cell_type=${2}
cofine_simulation_dir=${3}
n_genes=${4}
seed=${5}

source ~/.bash_profile
conda activate plink_env
mkdir -p ${cofine_simulation_dir}

date

simulation_stem=${cell_type}_cofine_sim_n_genes_${n_genes}_seed_${seed}

# Simulate summary statistics from the CoFine model on the real LD of the first n_genes genes
python simulate_cofine_data.py --output_dir ${cofine_simulation_dir} --output_stem ${simulation_stem} --fm_input_summary_file ${fm_input_summary_file} --n_genes ${n_genes} --seed ${seed}

# Fit CoFine, and the same sampler without caQTL and peak-gene link data
python run_cofine.py --sumstat_summary_file ${cofine_simulation_dir}${simulation_stem}_fine_mapping_input_summary.txt --output_stem ${cofine_simulation_dir}${simulation_stem}_cofine --seed ${seed}
python run_cofine.py --sumstat_summary_file ${cofine_simulation_dir}${simulation_stem}_fine_mapping_input_summary.txt --output_stem ${cofine_simulation_dir}${simulation_stem}_cofine_no_chromatin --no_chromatin --seed ${seed}

# PIP calibration and power, P(H = 1) calibration, hyper-parameter recovery
python evaluate_cofine_simulation.py --simulation_stem ${cofine_simulation_dir}${simulation_stem} --results_stems ${cofine_simulation_dir}${simulation_stem}_cofine ${cofine_simulation_dir}${simulation_stem}_cofine_no_chromatin --labels CoFine CoFine_no_chromatin

date
