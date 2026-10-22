#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 1-09:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores): every gene's low-rank Q and Gram matrix Q'Q are held in memory


fm_input_summary_file=${1}
cell_type=${2}
cofine_results_dir=${3}

source ~/.bash_profile
conda activate plink_env
mkdir -p ${cofine_results_dir}

date

##########################
# CoFine (grant Aim 3B): eQTL, caQTL and peak-gene link data
##########################
python run_cofine.py --sumstat_summary_file ${fm_input_summary_file} --output_stem ${cofine_results_dir}${cell_type}_cofine

##########################
# Comparison: the same sampler without caQTL and peak-gene link data
##########################
python run_cofine.py --sumstat_summary_file ${fm_input_summary_file} --output_stem ${cofine_results_dir}${cell_type}_cofine_no_chromatin --no_chromatin

date
