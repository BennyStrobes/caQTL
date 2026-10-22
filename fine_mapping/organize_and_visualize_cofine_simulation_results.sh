#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=10G                        # Memory total in MiB (for all cores)


cofine_simulation_dir=$1
simulation_stem_prefix=$2   # e.g. CD4_T_cofine_sim_n_genes_500 (seeds follow as _seed_<seed>)
cofine_simulation_viz_dir=$3

source ~/.bash_profile
conda activate plink_env
mkdir -p ${cofine_simulation_viz_dir}

# Calibration, power and hyper-parameter tables across seeds
python organize_cofine_simulation_results.py --simulation_dir ${cofine_simulation_dir} --simulation_stem_prefix ${simulation_stem_prefix} --output_stem ${cofine_simulation_viz_dir}${simulation_stem_prefix}_organized

# pdf plots
Rscript visualize_cofine_simulation_results.R ${cofine_simulation_viz_dir}${simulation_stem_prefix}_organized ${cofine_simulation_viz_dir}
