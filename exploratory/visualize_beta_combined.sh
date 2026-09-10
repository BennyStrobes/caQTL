#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)



cell_type="${1}"
pred_beta_combined_dir="${2}"
visualization_dir="${3}"

source ~/.bash_profile
conda activate plink_env


combination_version="all_links"
beta_combined_file=${pred_beta_combined_dir}${cell_type}"_beta_combined_"${combination_version}".tsv.gz"
n_bins="50"
output_prefix=${visualization_dir}${cell_type}"_beta_combined_"${combination_version}
python visualize_beta_combined.py \
    --beta_combined_file ${beta_combined_file} \
    --cell_type ${cell_type} \
    --n_bins ${n_bins} \
    --output_prefix ${output_prefix}
