#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-1:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=10G                        # Memory total in MiB (for all cores)


fm_input_summary_file=$1   # the input of run_cofine.sh
cofine_results_dir=$2
cell_type=$3
cofine_results_viz_dir=$4

source ~/.bash_profile
conda activate plink_env
mkdir -p ${cofine_results_viz_dir}

# Per-gene eQTL counts, per-variant PIPs / P(H = 1) / eQTL z-scores, and hyper-parameter samples of both fits
python organize_cofine_real_data_results.py --sumstat_summary_file ${fm_input_summary_file} --cofine_stem ${cofine_results_dir}${cell_type}_cofine --no_chromatin_stem ${cofine_results_dir}${cell_type}_cofine_no_chromatin --output_stem ${cofine_results_viz_dir}${cell_type}_cofine_organized

# pdf plots
Rscript visualize_cofine_real_data_results.R ${cofine_results_viz_dir}${cell_type}_cofine_organized ${cofine_results_viz_dir} ${cell_type}
