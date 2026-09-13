#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=30G                        # Memory total in MiB (for all cores)


simulation_fine_mapping_results_dir=$1
sumstats_output_dir=$2
n_simulations=$3
eqtl_sample_size=$4
organized_results_file=$5
viz_dir=$6


source ~/.bash_profile
conda activate plink_env


# Merge truth and the PIPs of both methods across simulations into one variant-level table
python organize_fine_mapping_simulation_results.py \
    --simulation_fine_mapping_results_dir ${simulation_fine_mapping_results_dir} \
    --sumstats_output_dir ${sumstats_output_dir} \
    --n_simulations ${n_simulations} \
    --eqtl_sample_size ${eqtl_sample_size} \
    --output_file ${organized_results_file}

# Calibration and power plots across simulations
Rscript visualize_fine_mapping_simulation_results.R ${organized_results_file} ${viz_dir}
