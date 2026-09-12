#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-2:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)


simulation_number=$1
gene_summary_file=$2
sumstats_output_dir=$3
simulation_fine_mapping_results_dir=$4
eqtl_sample_size=$5



source ~/.bash_profile
conda activate plink_env


# Stem to be used for output files
simulation_stem="simulation_${simulation_number}_eqtl_sample_size_${eqtl_sample_size}"



##########################
# 1. Simulate the data and the summary statistics
##########################
if false; then
python simulate_data_and_generate_sumstats.py ${simulation_number} ${gene_summary_file} ${sumstats_output_dir} ${simulation_stem}
fi

# Cross-gene summary files produced by simulate_data_and_generate_sumstats.py
sim_causal_effect_summary_file="${sumstats_output_dir}${simulation_stem}_simulated_causal_effect_summary.txt"
sim_sumstat_summary_file="${sumstats_output_dir}${simulation_stem}_sumstats_summary.txt"

##########################
# 2. Fun standard, eqtl ONLY fine-mapping
##########################
eqtl_only_finemapping_output_file="${simulation_fine_mapping_results_dir}${simulation_stem}_eqtl_only_finemapping_results.txt"
python run_standard_susie_finemapping_of_eqtls.py ${sim_sumstat_summary_file} ${eqtl_only_finemapping_output_file} ${eqtl_sample_size}
